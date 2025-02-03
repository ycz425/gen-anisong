from typing import Optional, Union
from transformers import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class CustomTrainer(Trainer):
    def __init__(self, *args, eval_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_collator = self.data_collator
        self.eval_collator = eval_collator

    def get_train_dataloader(self) -> DataLoader:
        print('train!')
        self.data_collator = self.train_collator
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        print('eval!')
        self.data_collator = self.eval_collator
        return super().get_eval_dataloader(eval_dataset)
    
    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        self.data_collator = self.eval_collator
        return super().get_test_dataloader(test_dataset)