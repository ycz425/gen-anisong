from transformers import DefaultDataCollator
from models.third_party.midi_tokenizer import MIDITokenizerV2
from torch.nn.utils.rnn import pad_sequence
import torch

class MIDIDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer: MIDITokenizerV2, train: bool = True):
        super().__init__(return_tensors='pt')
        self.tokenizer = tokenizer
        self.train = train

    def __call__(self, batch):
        if self.train:
            batch = [torch.tensor(self.tokenizer.augment(example['x'])) for example in batch]
        else:
            batch = [torch.tensor(example['x']) for example in batch]
        batch = pad_sequence(batch, batch_first=True, padding_value=self.tokenizer.pad_id)
        
        return {'x': batch[:, :-1].contiguous(), 'y': batch[:, 1:].contiguous()}
