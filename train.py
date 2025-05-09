import os
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.optim import AdamW
from model import GPT, GPTConfig
from data import get_dataloader
from utils import save_checkpoint, load_checkpoint

# Hyperparameters
DATA_PATH = 'data/chat.txt'
BLOCK_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5
LR = 3e-4

def train_loop(rank, flags):
    device = xm.xla_device()

    # Data
    train_loader, tokenizer = get_dataloader(DATA_PATH, BLOCK_SIZE, BATCH_SIZE)
    train_loader = pl.MpDeviceLoader(train_loader, device)

    # Model
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
    )
    model = GPT(config).to(device)
    optimizer = AdamW(model.parameters(), lr=LR)

    # Optionally resume
    start_epoch = load_checkpoint(model, optimizer)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            logits, loss = model(x.to(device), y.to(device))
            loss.backward()
            xm.optimizer_step(optimizer)
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item():.4f}")
        avg_loss = total_loss / (batch_idx+1)
        print(f"Epoch {epoch} Average Loss {avg_loss:.4f}")
        save_checkpoint(model, optimizer, epoch+1)

if __name__ == '__main__':
    xmp.spawn(train_loop, args=({},), nprocs=8, start_method='fork')
