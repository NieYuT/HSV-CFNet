import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchmetrics.functional import structural_similarity_index_measure
from torch.cuda.amp.grad_scaler import GradScaler

from model import LYT
from losses import OptimizedCombinedLoss, PSNRSSIMOptimizedLoss
from dataloader import create_dataloaders
import os
import numpy as np
import re

def calculate_psnr(img1, img2, max_pixel_value=1.0):
    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2, max_pixel_value=1.0):
    try:
        ssim_val = structural_similarity_index_measure(img1, img2, data_range=max_pixel_value)
        return ssim_val.item()
    except:
        return 0.0

def validate(model, dataloader, device):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    with torch.no_grad():
        for low, high in dataloader:
            low, high = low.to(device), high.to(device)
            output = model(low)

            psnr = calculate_psnr(output, high)
            total_psnr += psnr

            ssim = calculate_ssim(output, high)
            total_ssim += ssim

    avg_psnr = total_psnr / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    return avg_psnr, avg_ssim

def save_checkpoint(model, optimizer, scheduler, epoch, best_psnr, path='/mnt/sdb1/hjs/JCC/lyt1/PyTorch/checkpoint_lightweight.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'best_psnr': best_psnr
    }
    torch.save(checkpoint, path)

def main():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    
    train_low = '/mnt/sdb1/hjs/JCC/lyt1/PyTorch/data/LOLv2/Real_captured/Train/Low'
    train_high = '/mnt/sdb1/hjs/JCC/lyt1/PyTorch/data/LOLv2/Real_captured/Train/Normal'
    test_low = '/mnt/sdb1/hjs/JCC/lyt1/PyTorch/data/LOLv2/Real_captured/Test/Low'
    test_high = '/mnt/sdb1/hjs/JCC/lyt1/PyTorch/data/LOLv2/Real_captured/Test/Normal'
    learning_rate = 1.5e-4  
    num_epochs = 3000   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'LR: {learning_rate}; Epochs: {num_epochs}')

    train_loader, test_loader = create_dataloaders(train_low, train_high, test_low, test_high, crop_size=256, batch_size=1)  
    
    if train_loader is None or test_loader is None:
        print("Error: Failed to create data loaders")
        return
        
    print(f'Train loader: {len(train_loader)}; Test loader: {len(test_loader)}')
    model = LYT(filters=96)  
    model = model.to(device)

    criterion = PSNRSSIMOptimizedLoss(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-5)  
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-7) 
    
    if torch.cuda.is_available():
        scaler = GradScaler()
    else:
        scaler = None

    start_epoch = 0
    best_psnr = 0
    checkpoint_path = '/mnt/sdb1/hjs/JCC/lyt1/PyTorch/checkpoint_lightweight.pth'
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_psnr = checkpoint['best_psnr']
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, best PSNR: {best_psnr:.6f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (low, high) in enumerate(train_loader):
            low, high = low.to(device), high.to(device)
            
            optimizer.zero_grad()

            output = model(low)
            loss = criterion(high, output)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  Warning: Loss is {loss.item()}")
                continue
            
            if not loss.requires_grad:
                print("  Warning: Loss does not require grad!")
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
                optimizer.step()

            train_loss += loss.item()
            

            if batch_idx == 0:
                print(f"  Batch 0 Loss: {loss.item():.6f}")

        avg_train_loss = train_loss / len(train_loader)
        
        avg_psnr, avg_ssim = validate(model, test_loader, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, PSNR: {avg_psnr:.6f}, SSIM: {avg_ssim:.6f}')
        scheduler.step()  

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), '/mnt/sdb1/hjs/JCC/lyt1/PyTorch/best_model_lightweight.pth')
            print(f'Saving lightweight model with PSNR: {best_psnr:.6f}')
            
        if avg_psnr > 27 and avg_ssim > 0.90:
            print(f"达到目标性能！PSNR: {avg_psnr:.6f}, SSIM: {avg_ssim:.6f}")
            break
            
        save_checkpoint(model, optimizer, scheduler, epoch, best_psnr, path=checkpoint_path)

if __name__ == '__main__':
    main() 