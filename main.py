from src.custom_midi_model import CustomMIDIModel
from src.model_run import train_midi_model, generate
from src.preprocess import create_datasets
from transformers import TrainingArguments


model = CustomMIDIModel.from_pretrained("skytnt/midi-model-tv2o-medium")
train_dataset, eval_dataset = create_datasets('data/fonzi', model.tokenizer, 0.15, 3000)
args = TrainingArguments(
    output_dir='checkpoints/trial_5',
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=28,
    lr_scheduler_type='constant',
    learning_rate=0.00002,
    weight_decay=0.0002,
    eval_strategy='epoch',
    eval_on_start=True,
    num_train_epochs=30,
    logging_strategy='epoch',
    max_grad_norm=1.0,
    bf16=True,
    save_strategy='best',
    metric_for_best_model='eval_loss',
    save_total_limit=1
)

print(model)
if __name__ == '__main__':
    train_midi_model(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model=model,
        net_layers=4,
        net_token_layers=3,
        args=args
    )
    pass
    # generate(
    #     checkpoint_path="skytnt/midi-model-tv2o-medium",
    #     save_path='results/test0.mid',
    #     max_len=2048
    # )
