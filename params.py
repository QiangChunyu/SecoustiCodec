import numpy as np


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
      super(AttrDict, self).__init__(*args, **kwargs)
      self.__dict__ = self

  def override(self, attrs):
    if isinstance(attrs, dict):
      self.__dict__.update(**attrs)
    elif isinstance(attrs, (list, tuple, set)):
      for attr in attrs:
        self.override(attr)
    elif attrs is not None:
      raise NotImplementedError
    return self



params = AttrDict(
    # Data params
    sample_rate=22050,
    n_mels=80, # mel维度
    prompt_feat_length=300, # 随机截取prompt mel 长度
    pad_val = 1024,

    # AudioEncoder params
    n_ctx = 15000,
    n_state = 512,
    n_head = 8,
    n_layer = 6,
    vq_num_units = 8192,
    vq_beta = 1.0,    
    # 3^8
    fsq_levels = [3,3,3,3,3,3,3,3],

    # Model params
    embed_dim = 512,

    # PostNet params
    pon_out_channels = 256,
    pon_kernel_size = 5,

    # transformer params
    m_num_layers = 6,
    m_d_model = 256,
    m_num_heads = 8,
    m_dim_feedforward = 256,
    m_dropout = 0.1,
    #split_value = -2,
    m_batch_first = True,

    # Streaming
    _seanet_kwargs = { "channels": 80, "dimension": 512, "causal": True, "n_filters": 64, "n_residual_layers": 1, 
                "activation": "ELU", "compress": 2, "dilation_base": 2, "disable_norm_outer_blocks": 0, "kernel_size": 7, 
                "residual_kernel_size": 3, "last_kernel_size": 3, "norm": "none", "pad_mode": "constant", "ratios": [2, 2], "true_skip": True
        },

    _encoder_transformer_kwargs = {
                "d_model":512, "num_heads": 8, "num_layers": 8, "causal": True, "layer_scale": 0.01,
                "context": 250, "conv_layout": True, "max_period": 10000, "gating": "none", "norm": "layer_norm", "positional_embedding": "rope",
                "dim_feedforward": 2048, "input_dimension": 512, "output_dimensions": [512],
            },
    _decoder_transformer_kwargs = {
                "d_model":512, "num_heads": 8, "num_layers": 8, "causal": True, "layer_scale": 0.01,
                "context": 250, "conv_layout": True, "max_period": 10000, "gating": "none", "norm": "layer_norm", "positional_embedding": "rope",
                "dim_feedforward": 2048, "input_dimension": 512, "output_dimensions": [512],
            },
    _semantic_transformer_kwargs = {
                "d_model":512, "num_heads": 8, "num_layers": 8, "causal": True, "layer_scale": 0.01,
                "context": 250, "conv_layout": True, "max_period": 10000, "gating": "none", "norm": "layer_norm", "positional_embedding": "rope",
                "dim_feedforward": 2048, "input_dimension": 512, "output_dimensions": [512],
            },
    _semantic_connector_kwargs = {
                "d_model":512, "num_heads": 8, "num_layers": 8, "causal": True, "layer_scale": 0.01,
                "context": 250, "conv_layout": True, "max_period": 10000, "gating": "none", "norm": "layer_norm", "positional_embedding": "rope",
                "dim_feedforward": 2048, "input_dimension": 512, "output_dimensions": [512],
            },
)
