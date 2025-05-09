import os
from transformers import GPT2TokenizerFast
from torch.utils.data import Dataset
import torch

class ChatDataset(Dataset):
    def __init__(self, data_path, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        self.examples = []
        for convo in lines:
            tokens = tokenizer(convo).input_ids
            for i in range(0, len(tokens) - block_size, block_size):
                self.examples.append((tokens[i:i+block_size], tokens[i+1:i+block_size+1]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x, y = self.examples[idx]
        return torch.tensor(x), torch.tensor(y)

def get_dataloader(data_path, block_size, batch_size):
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    dataset = ChatDataset(data_path, tokenizer, block_size)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True), tokenizer
