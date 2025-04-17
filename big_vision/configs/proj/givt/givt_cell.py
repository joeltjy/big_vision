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

# pylint: disable=line-too-long
r"""Train a GIVT encoder-decoder model on cell dataset."""

import itertools
import ml_collections
from big_vision.datasets.cell.cell_config import N_TIMESTAMPS, N_FEATURES

ConfigDict = ml_collections.ConfigDict

VTT_MODELS = {
    'base': dict(num_layers=12, num_decoder_layers=12, num_heads=12, mlp_dim=3072, emb_dim=768),
    'large': dict(num_layers=24, num_decoder_layers=24, num_heads=16, mlp_dim=4096, emb_dim=1024),
}

def get_config(runlocal=False):
  """Config for training."""
  config = ConfigDict()

  config.input = {}
  config.input.pp = (
      f'keep("image")|'
      f'reshape(({N_TIMESTAMPS}, {N_FEATURES}, 1))'
  )
  pp_eval = (
      f'keep("image")|'
      f'reshape(({N_TIMESTAMPS}, {N_FEATURES}, 1))'
  )
  pp_pred = (
      f'keep("image")|'
      f'reshape(({N_TIMESTAMPS}, {N_FEATURES}, 1))'
  )
  
  config.input.data = dict(name='cell', split='train')
  config.input.batch_size = 1
  config.input.shuffle_buffer_size = 1000

  config.total_epochs = 200

  config.log_training_steps = 50
  config.ckpt_steps = 1000
  config.keep_ckpt_steps = None
  config.prefetch_to_device = 2
  config.seed = 0

  # Optimizer section
  config.optax_name = 'big_vision.scale_by_adafactor'
  config.optax = dict(beta2_cap=0.95)

  config.ar_generation_config = ml_collections.ConfigDict()
  config.ar_generation_config.temp = 0.85
  config.ar_generation_config.temp_probs = 1.0
  config.ar_generation_config.beam_size = 4
  config.ar_generation_config.fan_size = 8
  config.ar_generation_config.rand_top_k = False
  config.ar_generation_config.rand_top_k_temp = 1.0

  config.lr = 0.001
  config.wd = 0.000001
  config.lr_mults = [
      ('pos_embedding_encoder.*', 0.1),
      ('EmbedPatches.*', 0.1),
      ('encoder.*', 0.1),
      ('decoder.*', 1.0)
  ]
  config.schedule = dict(decay_type='cosine', warmup_steps=4_000)

  # Oracle section
  config.vae = ConfigDict()
  config.vae.model_name = 'proj.givt.vit'
  config.vae.model = ConfigDict()
  config.vae.model.input_size = (N_TIMESTAMPS, N_FEATURES)
  config.vae.model.patch_size = (1, 1)
  config.vae.model.code_len = 256
  config.vae.model.width = 768
  config.vae.model.enc_depth = 6
  config.vae.model.dec_depth = 12
  config.vae.model.mlp_dim = 3072
  config.vae.model.num_heads = 12
  config.vae.model.codeword_dim = 16
  config.vae.model.code_dropout = 'none'
  config.vae.model.bottleneck_resize = True
  # values: (channel index in source image, number of classes)
  config.vae.model.inout_specs = {
      'image': (0, N_FEATURES)
  }
  
  ### config.vae.model_init = TODO: add model init

  # Model section
  config.model_name = 'proj.givt.givt'
  config.model_init = {'encoder': 'howto-i21k-L/16'}
  config.model_load = dict(dont_load=('cls', 'head/bias', 'head/kernel'))
  config.model = ConfigDict(VTT_MODELS['large'])
  config.model.patches = (1, 1)
  config.model.input_size = (N_TIMESTAMPS, N_FEATURES)
  config.model.posemb_type = 'learn'
  config.model.seq_len = config.vae.model.code_len
  config.model.num_labels = None
  config.model.num_mixtures = 1
  config.model.fix_square_plus = True
  config.model.out_dim = config.vae.model.codeword_dim
  config.model.scale_tol = 1e-6
  config.model.dec_dropout_rate = 0.0

  # Evaluation section
  config.evals = {}
  config.evals.val = ConfigDict()
  config.evals.val.type = 'mean'
  config.evals.val.pred = 'validation'
  config.evals.val.data = dict(name=config.input.data.name, split='train[:4096]')
  config.evals.val.pp_fn = pp_eval
  config.evals.val.log_steps = 1000

  config.eval_only = False

  base = {
      'type': 'proj.givt.cell',
      'data': {**config.input.data},
      'pp_fn': pp_pred,
      'pred': 'predict_cell',
      'log_steps': 2000,
  }

  config.evals.cell_val = {**base}
  config.evals.cell_val.data.split = 'validation'

  config.evals.save_pred = dict(type='proj.givt.save_predictions')
  config.evals.save_pred.pred = 'predict_cell'
  config.evals.save_pred.pp_fn = pp_eval
  config.evals.save_pred.log_steps = 100_000
  config.evals.save_pred.data = dict(config.input.data)
  config.evals.save_pred.data.split = 'validation[:1024]'
  config.evals.save_pred.outfile = 'inference.npz'

  if runlocal:
    config.input.batch_size = 4
    config.input.shuffle_buffer_size = 10
    config.evals.val.data.split = 'train[:16]'
    config.evals.val.log_steps = 20
    config.model.num_layers = 1
    config.model.num_decoder_layers = 1
    del config.model_init
    config.evals.val.data.split = 'validation[:4]'
    config.evals.cell_val.data.split = 'validation[:4]'
    config.evals.save_pred.data.split = 'validation[:4]'

  return config
