import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ms_ssim
from pytorch_msssim import ssim
import pywt

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        self.loss_model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU()
        ).to(device).eval()
        for p in self.loss_model.parameters():
            p.requires_grad = False

    def forward(self, y_true, y_pred):
        device = next(self.loss_model.parameters()).device
        y_true = y_true.to(device)
        y_pred = y_pred.to(device)
        feat_true = self.loss_model(y_true)
        feat_pred = self.loss_model(y_pred)
        return F.mse_loss(feat_true, feat_pred)

def psnr_loss(y_true, y_pred):
    mse = F.mse_loss(y_true, y_pred)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
    return torch.clamp(40.0 - psnr, min=0.0)

def smooth_l1_loss(y_true, y_pred):
    return F.smooth_l1_loss(y_true, y_pred)

def custom_ms_ssim(y_true, y_pred, max_val=1.0, levels=3):
    weights = [0.0448, 0.2856, 0.3001][:levels] 
    weights = [w / sum(weights) for w in weights] 

    ssim_values = []
    for i in range(levels):
        current_ssim = ssim(y_true, y_pred, data_range=max_val, size_average=False)
        ssim_values.append(current_ssim)

        if i < levels - 1:
            y_true = F.avg_pool2d(y_true, kernel_size=2)
            y_pred = F.avg_pool2d(y_pred, kernel_size=2)

    ms_ssim_val = torch.ones_like(ssim_values[0])
    for w, s in zip(weights, ssim_values):
        ms_ssim_val *= s ** w

    return ms_ssim_val.mean()

def multiscale_ssim_loss(y_true, y_pred, max_val=1.0):
    return 1.0 - custom_ms_ssim(y_true, y_pred, max_val)

def color_loss(y_true, y_pred):
    mu_true = torch.mean(y_true, dim=[1,2,3])
    mu_pred = torch.mean(y_pred, dim=[1,2,3])
    return torch.mean(torch.abs(mu_true - mu_pred))

def hue_loss_circular(y_true_h, y_pred_h):
    diff = torch.abs((y_true_h - y_pred_h + 0.5) % 1.0 - 0.5)
    return torch.mean(diff)

def rgb_to_hsv_tensor(x):
    r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]
    maxc, _ = torch.max(x, dim=1, keepdim=True)
    minc, _ = torch.min(x, dim=1, keepdim=True)
    delta = maxc - minc + 1e-6

    v = maxc
    s = delta / (maxc + 1e-6)
    h = torch.zeros_like(maxc)

    mask = (maxc == r)
    h[mask] = ((g - b)[mask] / delta[mask]) % 6
    mask = (maxc == g)
    h[mask] = ((b - r)[mask] / delta[mask] + 2)
    mask = (maxc == b)
    h[mask] = ((r - g)[mask] / delta[mask] + 4)

    h = h / 6.0
    return h, s, v

def hsv_loss(y_true, y_pred):
    h_true, s_true, v_true = rgb_to_hsv_tensor(y_true)
    h_pred, s_pred, v_pred = rgb_to_hsv_tensor(y_pred)
    
    h_loss = hue_loss_circular(h_true, h_pred)
    
    s_loss = F.mse_loss(s_true, s_pred)
    
    v_loss = F.mse_loss(v_true, v_pred)
    
    return h_loss, s_loss, v_loss


def multiscale_consistency_loss(y_true, y_pred, scales=[1, 2, 4]):
    total_loss = 0.0
    
    for scale in scales:
        if scale == 1:
            true_scaled = y_true
            pred_scaled = y_pred
        else:
            true_scaled = F.avg_pool2d(y_true, kernel_size=scale)
            pred_scaled = F.avg_pool2d(y_pred, kernel_size=scale)
        
        scale_loss = F.mse_loss(true_scaled, pred_scaled)
        total_loss += scale_loss / len(scales)
    
    return total_loss


class PSNRSSIMOptimizedLoss(nn.Module):
    def __init__(self, device):
        super(PSNRSSIMOptimizedLoss, self).__init__()
        self.perc_loss = VGGPerceptualLoss(device)

        self.alpha_mse = 3.5       
        self.alpha_l1 = 1.2        
        self.alpha_ssim = 4.5      
        self.alpha_ms_ssim = 3.5   
        self.alpha_perc = 0.05   
        self.alpha_hsv_v = 0.4   
        self.alpha_hsv_s = 0.15    
        self.alpha_hsv_h = 0.08     
        self.alpha_multiscale = 0.2 
        
    def forward(self, y_true, y_pred):
        mse_loss = F.mse_loss(y_true, y_pred)
        l1_loss = F.l1_loss(y_true, y_pred)

        try:
            perc_loss = self.perc_loss(y_true, y_pred)
        except:
            perc_loss = torch.tensor(0.0, device=y_true.device)
        
        try:
            ssim_val = ssim(y_true, y_pred, data_range=1.0, size_average=True)
            if isinstance(ssim_val, torch.Tensor):
                ssim_loss = 1.0 - ssim_val
            else:
                ssim_loss = torch.tensor(1.0 - float(ssim_val), device=y_true.device)
        except:
            ssim_loss = torch.tensor(0.0, device=y_true.device)
        
        try:
            ms_ssim_val = ms_ssim(y_true, y_pred, data_range=1.0, size_average=True)
            if isinstance(ms_ssim_val, torch.Tensor):
                ms_ssim_loss = 1.0 - ms_ssim_val
            else:
                ms_ssim_loss = torch.tensor(1.0 - float(ms_ssim_val), device=y_true.device)
        except:
            ms_ssim_loss = torch.tensor(0.0, device=y_true.device)
        
        try:
            h_loss, s_loss_hsv, v_loss = hsv_loss(y_true, y_pred)
        except:
            h_loss = s_loss_hsv = v_loss = torch.tensor(0.0, device=y_true.device)
        
    
        
        try:
            multiscale_loss = multiscale_consistency_loss(y_true, y_pred)
        except:
            multiscale_loss = torch.tensor(0.0, device=y_true.device)
        
        current_psnr = 20 * torch.log10(1.0 / torch.sqrt(mse_loss + 1e-8))
        current_ssim = 1.0 - ssim_loss
        
        psnr_val = current_psnr.item() if isinstance(current_psnr, torch.Tensor) else current_psnr
        ssim_val = current_ssim.item() if isinstance(current_ssim, torch.Tensor) else current_ssim
        
        if psnr_val < 25:
            psnr_weight = 1.5
        elif psnr_val > 30:
            psnr_weight = 0.8
        else:
            psnr_weight = 1.0
            
        if ssim_val < 0.85:
            ssim_weight = 1.5
        elif ssim_val > 0.95:
            ssim_weight = 0.8
        else:
            ssim_weight = 1.0
        
        total_loss = (
            psnr_weight * self.alpha_mse * mse_loss +       
            psnr_weight * self.alpha_l1 * l1_loss +        
            ssim_weight * self.alpha_ssim * ssim_loss +        
            ssim_weight * self.alpha_ms_ssim * ms_ssim_loss +  
            self.alpha_perc * perc_loss +                         
            self.alpha_hsv_v * v_loss +                        
            self.alpha_hsv_s * s_loss_hsv +                  
            self.alpha_hsv_h * h_loss +                          
            self.alpha_multiscale * multiscale_loss                  
        )
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: PSNRSSIMOptimizedLoss total is {total_loss.item()}")
            return torch.tensor(0.0, requires_grad=True, device=y_true.device)
        
        return total_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
psnr_ssim_optimized_loss = PSNRSSIMOptimizedLoss(device) 
