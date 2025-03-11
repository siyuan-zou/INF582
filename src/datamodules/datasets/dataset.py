from torch.utils.data import Dataset
import pandas as pd
import os
import torch

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

class TextSummaryDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        """
        Args:
            data (pandas.DataFrame): 包含 'text' 和 'summary' 两列的数据
            tokenizer: 预训练 tokenizer（如 BERT, T5）
            max_length: 最大 token 长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.collate_fn = collate_fn

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        summary = self.data.iloc[idx]['summary']

        # Tokenize 并转换为模型输入格式
        text_enc = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        summary_enc = self.tokenizer(
            summary,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Handle padding in labels to avoid loss computation on padding tokens
        labels = summary_enc['input_ids'].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': text_enc['input_ids'].squeeze(0),
            'attention_mask': text_enc['attention_mask'].squeeze(0),
            'labels': labels  # target summary as labels
        }