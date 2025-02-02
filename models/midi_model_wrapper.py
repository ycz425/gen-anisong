from models.third_party.midi_model import MIDIModel
from transformers import PreTrainedModel
import torch.nn as nn


class MIDIModelWrapper(PreTrainedModel):
    def __init__(self, midi_model: MIDIModel):
        super().__init__(midi_model.config)
        self.midi_model = midi_model
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, cache=None):
        y = x[:, 1:].contiguous()
        x = x[:, :-1].contiguous()  # (batch_size, midi_sequence_length, token_sequence_length)
        hidden = self.midi_model.forward(x)
        print(hidden)
        # hidden = hidden.reshape(-1, hidden.shape[-1])
        # y = y.reshape(-1, y.shape[-1])  # (batch_size*midi_sequence_length, token_sequence_length)
        # x = y[:, :-1]
        # logits = self.midi_model.forward_token(hidden, x)
        # loss = self.loss_fn(
        #     logits.view(-1, self.midi_model.tokenizer.vocab_size),
        #     y.view(-1),
        #     reduction="mean",
        #     ignore_index=self.midi_model.tokenizer.pad_id
        # )
        # self.log("train/loss", loss)
        # # self.log("train/lr", self.lr_schedulers().get_last_lr()[0])
        # return loss, logits
        return None
