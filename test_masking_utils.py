import unittest

from transformers.testing_utils import is_torch_available, require_torch

if is_torch_available():
    import torch
    from torch.nn.attention.flex_attention import create_block_mask

    from transformers import LlamaConfig
    from transformers.masking_utils import create_causal_mask, find_packed_sequence_indices

config = LlamaConfig()
config._attn_implementation = "flex_attention"

batch_size = 2
sequence_length = 10
cache_position = torch.arange(sequence_length)

# First batch has 3 packed sequences of 4, 2 and 4 tokens respectively, second has 2 of 6 and 4 tokens
position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]])

causal_mask = create_causal_mask(
    config=config,
    # we only need batch size, seq_length and dtype here - we don't care about the values of the embeddings
    input_embeds=torch.empty((batch_size, sequence_length), dtype=torch.float16),
    attention_mask=None,
    cache_position=cache_position,
    past_key_values=None,
    position_ids=position_ids,
)

print(causal_mask.to_string())