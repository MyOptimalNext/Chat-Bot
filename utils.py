import os
import torch

def save_checkpoint(model, optimizer, epoch, path='checkpoint.pt'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'epoch': epoch
    }, path)

def load_checkpoint(model, optimizer, path='checkpoint.pt'):
    if os.path.exists(path):
        ckpt = torch.load(path)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        print(f"Resumed from epoch {ckpt['epoch']}")
        return ckpt['epoch']
    return 0
