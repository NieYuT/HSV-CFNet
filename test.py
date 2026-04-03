import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from model import LYT
from dataloader import create_dataloaders
import os
from torchvision.utils import save_image
from torchmetrics.functional import structural_similarity_index_measure
import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from model import LYT
from dataloader import create_dataloaders
import os
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import glob

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

def calculate_lpips(img1, img2, net_type='alex'):

    device = img1.device
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type=net_type).to(device)
    lpips_val = lpips_metric(img1, img2)
    return lpips_val.item()

def validate(model, dataloader, device, result_dir):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    with torch.no_grad():
        for idx, (low, high) in enumerate(dataloader):
            low, high = low.to(device), high.to(device)
            output = model(low)
            output = torch.clamp(output, 0, 1)

            # Save the output image
            save_image(output, os.path.join(result_dir, f'result_{idx}.png'))

            # Calculate PSNR
            psnr = calculate_psnr(output, high)
            total_psnr += psnr

            # Calculate SSIM
            ssim = calculate_ssim(output, high)
            total_ssim += ssim

            # Calculate LPIPS
            lpips = calculate_lpips(output, high)
            total_lpips += lpips

    avg_psnr = total_psnr / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    avg_lpips = total_lpips / len(dataloader)
    return avg_psnr, avg_ssim, avg_lpips

def main():
    # Paths and device setup
    test_low = '/mnt/sdb1/hjs/JCC/lyt1/PyTorch/data/LOLv1/Test/input'
    test_high = '/mnt/sdb1/hjs/JCC/lyt1/PyTorch/data/LOLv1/Test/target'
    weights_path = '/mnt/sdb1/hjs/JCC/lyt1/PyTorch/best_model_lightweight.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_name = test_low.split('/')[1]
    result_dir = os.path.join('results', dataset_name)
    os.makedirs(result_dir, exist_ok=True)

    _, test_loader = create_dataloaders(None, None, test_low, test_high, crop_size=None, batch_size=4)
    print(f'Test loader: {len(test_loader)}')

    model = LYT(filters=96).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f'Model loaded from {weights_path}')

    avg_psnr, avg_ssim, avg_lpips = validate(model, test_loader, device, result_dir)
    print(f'Validation PSNR: {avg_psnr:.6f}, SSIM: {avg_ssim:.6f}, LPIPS: {avg_lpips:.6f}')

if __name__ == '__main__':
    main()
