from typing import Any
from transformers import TrainingArguments
from datasets import Dataset
from src.custom_trainer import CustomTrainer
from src.custom_midi_model import CustomMIDIModel
from src.preprocess import create_datasets
from src.midi_data_collator import MIDIDataCollator


def train_midi_model(model: CustomMIDIModel, train_dataset: Dataset, eval_dataset: Dataset, net_layers: int, net_token_layers: int, args: TrainingArguments) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = True
    for param in model.net.layers[-net_layers:].parameters():
        param.requires_grad = True
    for param in model.net_token.layers[-net_token_layers:].parameters():
        param.requires_grad = True

    train_collator = MIDIDataCollator(model.tokenizer, train=True)
    eval_collator = MIDIDataCollator(model.tokenizer, train=False)

    trainer = CustomTrainer(
        model,
        args=args,
        data_collator=train_collator,
        eval_collator=eval_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    trainer.train()


def generate(checkpoint_path: str, save_path: str, prompt: Any = None, max_len: int = 512, temp: float = 1, top_p: float = 0.98, top_k: int = 20) -> None:
    model = CustomMIDIModel.from_pretrained(checkpoint_path)
    sequence = model.generate(
            prompt=prompt,
            batch_size=1,
            max_len=max_len,
            temp=temp,
            top_p=top_p,
            top_k=top_k
    )[0]
    from third_party import MIDI
    detokenized = model.tokenizer.detokenize(sequence)
    midi = MIDI.score2midi(detokenized)
    with open(save_path, 'wb') as f:
        f.write(midi)