from third_party.midi_model import MIDIModel, MIDIModelConfig
import torch.nn.functional as F
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, LlamaConfig


class LlamaAttentionWithDropout(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int, p: float):
        super().__init__(config, layer_idx)
        self.dropout = nn.Dropout(p)

    def forward(self, *args, **kwargs):
        attn_output, attn_weights = super().forward(*args, **kwargs)
        return self.dropout(attn_output), attn_weights
    

class LlamaMLPWithDropout(LlamaMLP):
    def __init__(self, config, p: float):
        super().__init__(config)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        down_proj = self.down_proj(self.dropout(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))
        return down_proj


class CustomMIDIModel(MIDIModel):
    def __init__(self, config: MIDIModelConfig):
        super().__init__(config)

        for layer in self.net.layers:
            layer.self_attn = LlamaAttentionWithDropout(layer.self_attn.config, layer.self_attn.layer_idx, p=0.2)
            layer.mlp = LlamaMLPWithDropout(layer.mlp.config, p=0.2)

        for layer in self.net_token.layers:
            layer.self_attn = LlamaAttentionWithDropout(layer.self_attn.config, layer.self_attn.layer_idx, p=0.2)
            layer.mlp = LlamaMLPWithDropout(layer.mlp.config, p=0.2)


    def forward(self, x, labels=None, cache=None):
        # print(f"Allocated memory: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
        # print(x.shape)
        if labels is None:  # for generation: redirect to superclass forward
            return super().forward(x, cache=cache)
        else:
            hidden = super().forward(x, cache=cache)
            hidden = hidden.reshape(-1, hidden.shape[-1])
            labels = labels.reshape(-1, labels.shape[-1])  # (batch_size*midi_sequence_length, token_sequence_length)
            x = labels[:, :-1]
            logits = super().forward_token(hidden, x)
            loss = F.cross_entropy(
                logits.view(-1, self.tokenizer.vocab_size),
                labels.view(-1),
                reduction="mean",
                ignore_index=self.tokenizer.pad_id
            )
            return {'loss': loss, 'logits': logits}
