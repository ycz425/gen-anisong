from ..models.third_party.midi_tokenizer import MIDITokenizerV2
from ..utils.third_party import MIDI
from datasets import Dataset
import random
import os

def process_midi(midi_file: str, tokenizer: MIDITokenizerV2):
    with open(midi_file, 'rb') as file:
        data = file.read()
    
    if len(data) > 384000 or len(data) < 3000:
        raise Exception()

    score = MIDI.midi2score(data)
    score = merge_tracks(score)
    tokenized_midi = tokenizer.tokenize(score)

    return tokenized_midi


def merge_tracks(score):
    output = [score[0], []]
    for track in score[1:]:
        output[1].extend(track)
    return output


def create_datasets(dataset_dir: str, tokenizer: MIDITokenizerV2, val_split: float):
    tokenized_midis = []

    for dirpath, _, filenames in os.walk(dataset_dir):
        for filename in filenames:
            if filename.endswith('.mid'):
                tokenized_midis.append(process_midi(f'{dirpath}/{filename}', tokenizer))

    random.shuffle(tokenized_midis)

    train_inputs = []
    train_targets = []
    eval_inputs = []
    eval_targets = []

    for i, midi in enumerate(tokenized_midis):
        if i < int(len(tokenized_midis) * val_split):
            train_inputs.append(midi[1:])
            train_targets.append(midi[:-1])
        else:
            eval_inputs.append(midi[1:])
            eval_targets.append(midi[:-1])
    
    train = Dataset.from_dict({'input_ids': train_inputs, 'labels': train_targets})
    eval = Dataset.from_dict({'input_ids': eval_inputs, 'labels': eval_targets})
    return train, eval, tokenized_midis