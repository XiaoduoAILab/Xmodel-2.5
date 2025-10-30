# Copyright (c) 2023 XiaoDuo AI. All rights reserved.

from typing import Optional, Dict

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging
from typing_extensions import Self

logger = logging.get_logger(__name__)


class XmodelConfig(PretrainedConfig):
    """Configuration class for Xmodel models.

    Args:
        vocab_size (int): Vocabulary size of the model
        hidden_size (int): Dimensionality of the embeddings and hidden states
        intermediate_size (int): Dimensionality of the "intermediate" (feed-forward) layer
        num_hidden_layers (int): Number of hidden layers in the Transformer encoder
        num_attention_heads (int): Number of attention heads for each attention layer
        num_key_value_heads (int): Number of key/value heads (for grouped query attention)
        hidden_act (str): Activation function for hidden layers
        max_position_embeddings (int): Maximum sequence length
        initializer_range (float): Standard deviation for weight initialization
        rms_norm_eps (float): Epsilon for RMS layer normalization
        use_cache (bool): Whether to use caching for faster generation
        rope_theta (float): Base frequency for rotary embeddings
        rope_scaling (Optional[Dict]): Scaling configuration for rotary embeddings
        attention_bias (bool): Whether to use bias in attention layers
        attention_dropout (float): Dropout probability for attention layers
        mlp_bias (bool): Whether to use bias in MLP layers
        scale_emb (float): Scaling factor for embeddings
        dim_model_base (int): Base dimension for model scaling
        scale_depth (float): Depth scaling factor
    """
    model_type = "xmodel"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            vocab_size: int = 32000,
            hidden_size: int = 1536,
            intermediate_size: Optional[int] = 4096,
            num_hidden_layers: int = 48,
            num_attention_heads: int = 24,
            num_key_value_heads: Optional[int] = 8,
            hidden_act: str = "silu",
            max_position_embeddings: int = 4096,
            initializer_range: float = 0.1,
            rms_norm_eps: float = 1e-5,
            use_cache: bool = True,
            pad_token_id: Optional[int] = None,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            pretraining_tp: int = 1,
            tie_word_embeddings: bool = True,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias: bool = False,
            attention_dropout: float = 0.0,
            mlp_bias: bool = False,
            use_mup: bool = False,
            mup_input_scale: float = 12.0,
            mup_output_scale: float = 1.0,
            mup_attention_residual_scale: float = 1.4,
            mup_ffn_residual_scale: float = 1.4,
            mup_base_width: int = 256,
            **kwargs,
    ):
        # Validate input parameters
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_hidden_layers <= 0:
            raise ValueError(f"num_hidden_layers must be positive, got {num_hidden_layers}")
        if num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {num_attention_heads}")
        if max_position_embeddings <= 0:
            raise ValueError(f"max_position_embeddings must be positive, got {max_position_embeddings}")

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size

        # Calculate intermediate size if not provided
        if intermediate_size is None:
            self.intermediate_size = find_multiple(int(8 * hidden_size / 3), 256)
        else:
            if intermediate_size <= 0:
                raise ValueError(f"intermediate_size must be positive, got {intermediate_size}")
            self.intermediate_size = intermediate_size

        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # Handle grouped query attention
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        elif num_key_value_heads > num_attention_heads:
            raise ValueError(
                f"num_key_value_heads ({num_key_value_heads}) must be <= num_attention_heads ({num_attention_heads})"
            )

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.use_mup = use_mup
        self.mup_input_scale = mup_input_scale
        self.mup_output_scale = mup_output_scale
        self.mup_attention_residual_scale = mup_attention_residual_scale
        self.mup_ffn_residual_scale = mup_ffn_residual_scale
        self.mup_base_width = mup_base_width
        self.mup_width_multiplier = hidden_size / mup_base_width
        self._attn_implementation = "eager"

        self.auto_map = {
            "AutoConfig": "configuration_xmodel2.XmodelConfig",
            "AutoModelForCausalLM": "modeling_xmodel2.XmodelForCausalLM"
        }

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def from_name(cls, name: str) -> Self:
        """Create a configuration from a preset model name.

        Args:
            name (str): Name of the model preset (e.g. 'nano', 'small', 'xl')

        Returns:
            XmodelConfig: Configuration instance

        Raises:
            ValueError: If the model name is not found in presets
        """
        if name not in xmodel_configs:
            raise ValueError(
                f"Unknown model name '{name}'. Available models: {list(xmodel_configs.keys())}"
            )
        return cls(**xmodel_configs[name])


xmodel_configs = {
    "nano": dict(num_hidden_layers=8,
                 num_attention_heads=4,
                 num_key_value_heads=1,
                 hidden_size=256,
                 tie_word_embeddings=True,
                 intermediate_size=640),

    "micro": dict(num_hidden_layers=12,
                  num_attention_heads=6,
                  num_key_value_heads=1,
                  hidden_size=384,
                  tie_word_embeddings=True,
                  intermediate_size=960),

    "tiny": dict(num_hidden_layers=18,
                 num_attention_heads=8,
                 num_key_value_heads=4,
                 hidden_size=512,
                 tie_word_embeddings=True,
                 intermediate_size=1280),

    # GPT-1 & Bert-Base
    "small": dict(num_hidden_layers=30,
                  num_attention_heads=9,
                  num_key_value_heads=3,
                  hidden_size=576,
                  tie_word_embeddings=True,
                  intermediate_size=1440),

    # Bert-Large
    "medium": dict(num_hidden_layers=32,
                   num_attention_heads=15,
                   num_key_value_heads=5,
                   hidden_size=960,
                   tie_word_embeddings=True,
                   intermediate_size=2400),

    # GPT-2
    "xl": dict(num_hidden_layers=48,
               num_attention_heads=24,
               num_key_value_heads=8,
               hidden_size=1536,
               tie_word_embeddings=True,
               intermediate_size=3840),  # GPT-2

    "2B": dict(num_hidden_layers=40,
               num_attention_heads=36,
               num_key_value_heads=6,
               hidden_size=2304,
               tie_word_embeddings=True,
               intermediate_size=5760),

    "4B": dict(num_hidden_layers=62,
               num_attention_heads=40,
               num_key_value_heads=8,
               hidden_size=2560,
               tie_word_embeddings=True,
               intermediate_size=6400),

}


def find_multiple(n: int, k: int) -> int:
    """Find the smallest multiple of k that is >= n.

    Args:
        n (int): Target number
        k (int): Multiple base

    Returns:
        int: Smallest multiple of k >= n

    Raises:
        ValueError: If k <= 0
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if n % k == 0:
        return n
    return n + k - (n % k)
