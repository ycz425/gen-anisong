from third_party.midi_tokenizer import MIDITokenizerV2
from third_party import MIDI
from datasets import Dataset
import random
import os

def process_midi(midi_file: str, tokenizer: MIDITokenizerV2):
    with open(midi_file, 'rb') as file:
        data = file.read()

    score = MIDI.midi2score(data)
    score = merge_tracks(score)
    
    score = remove_control_change_events(score)
    score = remove_empty_first_measure(score)
    tokenized_midi = tokenizer.tokenize(score)

    return tokenized_midi


def merge_tracks(score):
    output = [score[0], []]
    for track in score[1:]:
        output[1].extend(track)
    return output


def remove_empty_first_measure(score):
    """score must have only one track.
    """
    if len(score) > 2:
        raise ValueError('Multiple tracks found in score.')
    
    ticks, track = score
    time_signature = None

    for event in track:
        if event[0] == 'time_signature' and event[1] == 0:
            time_signature = (event[2], event[3])
            break

    if time_signature is None:
        raise ValueError('Initial time signature is missing.')

    n, d = time_signature[0], 2 ** time_signature[1]
    threshold = (4 / d * n - 0.125) * ticks

    while True:
        for event in track:
            if event[0] == 'note' and event[1] < threshold:
                return score
            elif event[0] == 'note':
                break

        for event in track:
            event[1] = max(event[1] - 4 / d * n * ticks, 0)


def remove_control_change_events(score):
    """score must have only one track.
    """
    new_score = [score[0], []]
    for event in score[1]:
        if event[0] != 'control_change':
            new_score[1].append(event)

    return new_score


def create_datasets(dataset_dir: str, tokenizer: MIDITokenizerV2, val_split: float, max_sequence_length: int = 3000) -> tuple[Dataset, Dataset]:
    tokenized_midis = []
    length = []

    for dirpath, _, filenames in os.walk(dataset_dir):
        for filename in filenames:
            if filename.endswith('.mid'):
                midi = process_midi(f'{dirpath}/{filename}', tokenizer)
                if not tokenizer.check_quality(midi):
                    print(f'{filename} ignored due to bad file quality.')
                elif len(midi) > max_sequence_length:
                    print(f'{filename} ignored due to exceeding max sequence length: {len(midi)} > {max_sequence_length}.')
                else:
                    tokenized_midis.append(midi)
                    length.append(len(midi))
                    

    print(f'Loaded {len(tokenized_midis)} midi files.')
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