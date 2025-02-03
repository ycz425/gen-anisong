from third_party.midi_tokenizer import MIDITokenizerV2
from third_party import MIDI
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
    eval_inputs = []

    for i, midi in enumerate(tokenized_midis):
        if i > int(len(tokenized_midis) * val_split):
            train_inputs.append(midi)
        else:
            eval_inputs.append(midi)
    
    train = Dataset.from_dict({'x': train_inputs, 'labels': [0] * len(train_inputs)})  # dummy 'labels' key to ensure Trainer will compute loss
    eval = Dataset.from_dict({'x': eval_inputs, 'labels': [0] * len(eval_inputs)})
    return train, eval