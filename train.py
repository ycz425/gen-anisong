from third_party.midi_model import MIDIModel
from transformers import TrainingArguments
from src.custom_trainer import CustomTrainer
from src.midi_model_wrapper import MIDIModelWrapper
from src.preprocess import create_datasets
from src.midi_data_collator import MIDIDataCollator


model = MIDIModel.from_pretrained("skytnt/midi-model-tv2o-medium")

for param in model.parameters():
    param.requires_grad = False
for param in model.lm_head.parameters():
    param.requires_grad = True
train_dataset, eval_dataset = create_datasets('data/fonzi', model.tokenizer, 0.1)

model = MIDIModelWrapper(model)

train_collator = MIDIDataCollator(model.midi_model.tokenizer, train=True)
eval_collator = MIDIDataCollator(model.midi_model.tokenizer, train=False)

args = TrainingArguments(
    output_dir='delete',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    prediction_loss_only=True,
    eval_strategy='epoch',
    num_train_epochs=5,
    logging_dir='./logs',
    logging_strategy='epoch',
    learning_rate=0.00005,
    lr_scheduler_type='linear',
    weight_decay=0.0001,
    max_grad_norm=1.0,
    save_strategy=''
)

trainer = CustomTrainer(
    model,
    args=args,
    data_collator=train_collator,
    eval_collator=eval_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)
trainer.train()
