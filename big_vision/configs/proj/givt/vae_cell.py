r"""Train VAE on cell dataset.
Uses ViT as encoder and transformer as decoder.
"""

import big_vision.configs.common as bvcc
import ml_collections as mlc
from big_vision.datasets.cell.cell_config import N_TIMESTAMPS, N_FEATURES
import torch.nn as nn

def get_config(arg=''):
  """Config for training VAE on cell dataset."""
  arg = bvcc.parse_arg(arg, runlocal=False, singlehost=False)
  config = mlc.ConfigDict()

  config.input = {}

  config.input.data = dict(name='cell', split='train')

  config.input.batch_size = 1
  config.input.shuffle_buffer_size = 1000

  config.total_epochs = 200

  # TODO: add pp for cell dataset
  config.input.pp = (
      f'keep("image")'
  )
  pp_eval = (
      f'keep("image")'
  )
  pp_pred = (
      f'keep("image")'
  )

  config.log_training_steps = 50
  config.ckpt_steps = 1000
  config.keep_ckpt_steps = None

  # Model section
  config.model_name = 'proj.givt.vit'
  config.model = mlc.ConfigDict()
  config.model.input_size = (1, N_FEATURES)  # Process one row
  config.model.patch_size = (1, 1)  # Process each element
  config.model.code_len = 256
  config.model.width = 768
  config.model.bottleneck_resize = False
  config.model.enc_depth = 6
  config.model.dec_depth = 12
  config.model.mlp_dim = 3072
  config.model.num_heads = 12
  config.model.codeword_dim = 16
  config.model.code_dropout = 'none'
  config.model.bottleneck_resize = True
  config.model.scan = True
  config.model.remat_policy = 'nothing_saveable'
  config.model_init = ''

  config.rec_loss_fn = 'l2' 
  config.mask_zero_target = True
 
  config.model.inout_specs = {
      'image': (0, N_FEATURES),  # Process each row independently
  }

  config.beta = 2e-4
  config.beta_percept = 0.0

  # Optimizer section
  config.optax_name = 'scale_by_adam'
  config.optax = dict(b2=0.95)

  # FSDP training by default
  config.sharding_strategy = [('.*', 'fsdp(axis="data")')]
  config.sharding_rules = [('act_batch', ('data',))]

  config.lr = 1e-3
  config.wd = 1e-4
  config.schedule = dict(decay_type='cosine', warmup_steps=0.1)
  config.grad_clip_norm = 1.0

  # Evaluation section
  config.evals = {}
  config.evals.val = mlc.ConfigDict()
  config.evals.val.type = 'mean'
  config.evals.val.pred = 'validation'
  config.evals.val.data = {**config.input.data}
  config.evals.val.data.split = 'validation'
  config.evals.val.pp_fn = pp_eval
  config.evals.val.log_steps = 250

  base = {
      'type': 'proj.givt.cell',
      'data': {**config.input.data},
      'pp_fn': pp_pred,
      'pred': 'predict_cell',
      'log_steps': 2000,
  }
  config.evals.cell_val = {**base}
  config.evals.cell_val.data.split = 'validation'

  # ### Uses a lot of memory
  # config.evals.save_pred = dict(type='proj.givt.save_predictions')
  # config.evals.save_pred.pp_fn = pp_eval
  # config.evals.save_pred.log_steps = 100_000
  # config.evals.save_pred.data = {**config.input.data}
  # config.evals.save_pred.data.split = 'validation[:64]'
  # config.evals.save_pred.batch_size = 64
  # config.evals.save_pred.outfile = 'inference.npz'

  config.eval_only = False
  config.seed = 0

  if arg.singlehost:
    config.input.batch_size = 128
    config.num_epochs = 50
  elif arg.runlocal:
    config.input.batch_size = 1
    config.input.shuffle_buffer_size = 10
    config.log_training_steps = 5
    config.model.enc_depth = 1
    config.model.dec_depth = 1
    config.evals.val.data.split = 'validation[:16]'
    config.evals.val.log_steps = 20
    config.evals.cell_val.data.split = 'validation[:16]'

  return config
