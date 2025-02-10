from third_party.midi_model import MIDIModel, MIDIModelConfig
import torch.nn.functional as F
import torch


class CustomMIDIModel(MIDIModel):
    def __init__(self, config: MIDIModelConfig):
        super().__init__(config)

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
