# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation for cell data."""

import functools
import itertools

from big_vision import input_pipeline
from big_vision import utils
from big_vision.datasets import core as ds_core
import big_vision.pp.builder as pp_builder
import jax
import jax.numpy as jnp
import numpy as np

# Temporary global flag to facilitate backwards compatability.
API = "jit"

@functools.cache
def _get_predict_fn(predict_fn, mesh=None):
    """Wrapper for jit-compiled predict function."""
    @functools.partial(jax.jit,
                     out_shardings=jax.sharding.NamedSharding(
                         mesh, jax.sharding.PartitionSpec()))
    def _run_predict_fn(train_state, batch):
        """Run predict_fn and gather all outputs on all devices."""
        pred = predict_fn(train_state, batch)
        return {
            "gt": batch["image"],
            "y": pred["logits"]["image"]
        }
    return _run_predict_fn

class Evaluator:
    """Evaluator for cell data."""

    def __init__(self,
                 predict_fn,
                 pp_fn,
                 batch_size,
                 data,
                 cache_final=True,
                 cache_raw=False,
                 prefetch=1,
                 *,
                 devices):
        """Evaluator for cell data.

        Args:
            predict_fn: jit-compilable function that outputs cell predictions
            pp_fn: Preprocessing function
            batch_size: Batch size
            data: Dict specifying name and split of the dataset
            cache_final: Whether to cache the data after preprocessing
            cache_raw: Whether to cache the raw data
            prefetch: Number of batches to prefetch
            devices: List of jax devices
        """
        self.predict_fn = _get_predict_fn(
            predict_fn, jax.sharding.Mesh(devices, ("devices",)))

        data = ds_core.get(**data)
        self.dataset, self.steps = input_pipeline.make_for_inference(
            data.get_tfdata(ordered=True), batch_size=batch_size,
            num_ex_per_process=data.num_examples_per_process(),
            preprocess_fn=pp_builder.get_preprocess_fn(pp_fn),
            cache_final=cache_final, cache_raw=cache_raw)
        self.data_iter = input_pipeline.start_global(
            self.dataset, devices, prefetch)

    def run(self, train_state):
        """Run cell data evaluation.

        Args:
            train_state: pytree containing the model parameters

        Yields:
            Tuples consisting of metric name and value
        """
        mses = []
        
        for batch in itertools.islice(self.data_iter, self.steps):
            out = self.predict_fn(train_state, batch)

            if jax.process_index():  # Host0 gets all preds and does eval
                continue

            out = jax.device_get(out)
            out = jax.tree_map(lambda x: x[out["mask"]], out)

            for gt, pred in zip(out["gt"], out["y"]):
                gt, pred = utils.put_cpu((gt, pred))
                gt, pred = np.array(gt), np.array(pred)
                
                # Calculate MSE
                mse = np.mean((gt - pred) ** 2)
                
                mses.append(mse)

        if jax.process_index():  
            return

        yield "MSE", np.mean(mses)
        