from models.third_party.midi_model import MIDIModel
from transformers import PreTrainedModel
import torch.nn.functional as F


class MIDIModelWrapper(PreTrainedModel):
    def __init__(self, midi_model: MIDIModel):
        super().__init__(midi_model.config)
        self.midi_model = midi_model
        self.loss_fn = F.cross_entropy

    def forward(self, x, labels, cache=None):
        hidden = self.midi_model.forward(x, cache)
        hidden = hidden.reshape(-1, hidden.shape[-1])
        labels = labels.reshape(-1, labels.shape[-1])  # (batch_size*midi_sequence_length, token_sequence_length)
        x = labels[:, :-1]
        logits = self.midi_model.forward_token(hidden, x)
        loss = self.loss_fn(
            logits.view(-1, self.midi_model.tokenizer.vocab_size),
            labels.view(-1),
            reduction="mean",
            ignore_index=self.midi_model.tokenizer.pad_id
        )
        # self.log("train/loss", loss)
        # self.log("train/lr", self.lr_schedulers().get_last_lr()[0])
        print(loss)
        return {'loss': loss, 'logits': logits}
