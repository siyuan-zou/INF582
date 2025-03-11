import os
import pandas as pd
from src.datamodules.datasets.dataset import TextSummaryDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import time



def load_data_from_folder(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            all_data.append(df)
    
    # 合并所有文件的数据
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data

class TextSummaryDataModule(LightningDataModule):
    def __init__(self, data_folder, tokenizer, batch_size=32, val_size=0.1, test_size=0.1):
        super().__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.tokenizer = tokenizer

    def prepare_data(self):
        # 加载数据
        data = load_data_from_folder(self.data_folder)
        train_data, temp_data = train_test_split(data, test_size=self.val_size + self.test_size, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=self.test_size / (self.val_size + self.test_size), random_state=42)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def setup(self, stage=None):
        # Split data into train, validation, and test sets
        """Setup the datamodule.
        Args:
            stage (str): stage of the datamodule
                Is be one of "fit" or "test" or None
        """
        print("Stage", stage)
        start_time = time.time()
        
        # Create dataset instances
        self.train_dataset = TextSummaryDataset(self.train_data, tokenizer=self.tokenizer)
        self.val_dataset = TextSummaryDataset(self.val_data, tokenizer=self.tokenizer)
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Val dataset size: {len(self.val_dataset)}")
        self.test_dataset = TextSummaryDataset(self.test_data, tokenizer=self.tokenizer)
        print(f"Test dataset size: {len(self.test_dataset)}")

        end_time = time.time()
        print(f"Setup took {(end_time - start_time):.2f} seconds")
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.train_dataset.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.val_dataset.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.test_dataset.collate_fn)