from src.model_run import train_midi_model, generate
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir='checkpoints/trial_4',
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
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
    save_total_limit=1,
)

if __name__ == '__main__':
    train_midi_model(net_layers=4, net_token_layers=3, args=args)