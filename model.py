import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import pywt
import numpy as np
# from flash_attn.modules.mha import FlashSelfAttention
from flash_attn.modules.mha import MHA
from einops import rearrange
from torch.nn import MultiheadAttention

class LayerNormalization(nn.Module):
    def __init__(self, dim):
        super(LayerNormalization, self).__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)
    

def dwt2(image):
    image_np = image.squeeze(0).cpu().numpy()
    coeffs2 = pywt.dwt2(image_np, 'haar') 
    LL, (LH, HL, HH) = coeffs2  


    LL = torch.tensor(LL).to(image.device)
    LH = torch.tensor(LH).to(image.device)
    HL = torch.tensor(HL).to(image.device)
    HH = torch.tensor(HH).to(image.device)
    return LL, LH, HL, HH


class SEBlock(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(input_channels, input_channels // reduction_ratio)
        self.fc2 = nn.Linear(input_channels // reduction_ratio, input_channels)
        self._init_weights()

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        y = self.pool(x).reshape(batch_size, num_channels)
        y = F.relu(self.fc1(y))
        y = torch.tanh(torch.clamp(self.fc2(y), -6, 6))
        y = y.reshape(batch_size, num_channels, 1, 1)
        return x * y
    
    def _init_weights(self):
        init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.fc1.bias, 0)
        init.constant_(self.fc2.bias, 0)


class SaturationAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.high_sat_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        self.low_sat_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        self.adaptive_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        high_attn = self.high_sat_branch(x)
        low_attn = self.low_sat_branch(x)
        weight = self.adaptive_weight(x)
        attn = weight * high_attn + (1 - weight) * low_attn
        
        return x * attn

class ValueAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gradient_extractor = SobelGradient()
        
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

        self.gradient_guide = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 1),  
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        gradient = self.gradient_extractor(x.mean(dim=1, keepdim=True))

        ca = self.channel_attn(x)
        x_channel = x * ca
        
        gradient_feat = torch.cat([x_channel, gradient], dim=1)
        guide_attn = self.gradient_guide(gradient_feat)
        
        return x_channel * guide_attn




class SpatialFrequencyJointAttention(nn.Module):
    def __init__(self, embed_dim=16):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(embed_dim, 1, 1),
            nn.Sigmoid()
        )

        self.frequency_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, C, H, W = x.shape

        spatial_weights = self.spatial_attention(x)  
        spatial_enhanced = x * spatial_weights
        
        freq_weights = self.frequency_attention(x) 
        freq_enhanced = x * freq_weights

        output = x + 0.5 * (spatial_enhanced + freq_enhanced)
        
        return output




class GradientGuidedEnhancement(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        self.base_channels = base_channels

        self.gradient_extractor = SobelGradient()

        self.adaptive_net = nn.Sequential(
            nn.Conv2d(2, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            ValueAttention(base_channels),  
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU()
        )
        
        self.spatial_weights = nn.Sequential(
            nn.Conv2d(base_channels, base_channels//4, 1),
            nn.ReLU(),
            nn.Conv2d(base_channels//4, 1, 1),
            nn.Sigmoid()
        )
        
        self.detail_branch = nn.Sequential(
            nn.Conv2d(base_channels, base_channels//2, 1),
            nn.ReLU(),
            nn.Conv2d(base_channels//2, base_channels, 1)
        )
        
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU()
        )
        
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            ValueAttention(base_channels), 
            nn.Conv2d(base_channels, 64, 3, padding=1)
        )
        
    def forward(self, v_channel):
        gradient = self.gradient_extractor(v_channel) 

        v_grad_input = torch.cat([v_channel, gradient], dim=1) 

        adaptive_features = self.adaptive_net(v_grad_input)  

        spatial_weights = self.spatial_weights(adaptive_features) 
        weighted_features = adaptive_features * spatial_weights  

        detail_features = self.detail_branch(adaptive_features)  

        fused_features = torch.cat([weighted_features, detail_features], dim=1)  
        enhanced_features = self.feature_fusion(fused_features) 

        output = self.output(enhanced_features)  
        
        return output

class CircularHProcessing(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        self.circular_embed = nn.Sequential(
            nn.Conv2d(2, base_channels, 3, padding=1), 
            nn.ReLU(),
            HueAttention(base_channels)  
        )

        self.cross_space_fusion = nn.Sequential(
            nn.Conv2d(base_channels + 1, base_channels, 3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU()
        )

        self.color_correction = nn.Sequential(
            nn.Conv2d(base_channels, base_channels//2, 1),
            nn.ReLU(),
            nn.Conv2d(base_channels//2, base_channels, 1),
            nn.Sigmoid()
        )

        self.output = nn.Conv2d(base_channels, 1, 3, padding=1)
        
        self._init_weights()
        
    def forward(self, h_channel):
        h_sin = torch.sin(2 * np.pi * h_channel)
        h_cos = torch.cos(2 * np.pi * h_channel)
        circular = torch.cat([h_sin, h_cos], dim=1) 
        
        features = self.circular_embed(circular)  

        fused = torch.cat([features, h_channel], dim=1)  
        enhanced = self.cross_space_fusion(fused) 
        
        correction = self.color_correction(enhanced)  
        enhanced = enhanced * correction
        output = self.output(enhanced)  
        return torch.clamp(output, 0, 1)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class TeacherNetwork(nn.Module):
    def __init__(self, embed_size=64, num_heads=8, feedforward_dim=256, use_gradient_guided=True):
        super(TeacherNetwork, self).__init__()
        self.embed_size = embed_size
        self.use_gradient_guided = use_gradient_guided
        
        if use_gradient_guided:
            self.gradient_enhancer = GradientGuidedEnhancement(base_channels=32)
      
        
    def forward(self, v_channel):
        if self.use_gradient_guided:
            v_output = self.gradient_enhancer(v_channel)
            return v_output


class ResidualQuantizationSProcessor(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        self.base_channels = base_channels
        

        self.base_extractor = nn.Sequential(
            nn.Conv2d(1, base_channels, 7, padding=3),  
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 7, padding=3),
            nn.ReLU(),
            SaturationAttention(base_channels)  
        )
        
        self.residual_extractor = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),  
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            SaturationAttention(base_channels)
        )

        self.weight_generator = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 1),
            nn.ReLU(),
            nn.Conv2d(base_channels, 2, 1),
            nn.Softmax(dim=1)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            SEBlock(base_channels, reduction_ratio=4)
        )
        
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            SaturationAttention(base_channels),  # S通道专用注意力
            nn.Conv2d(base_channels, 64, 3, padding=1)
        )
        
        self._init_weights()
    
    def forward(self, s):
        base_sat = self.base_extractor(s)  
        residual_sat = self.residual_extractor(s) 
        
        combined = torch.cat([base_sat, residual_sat], dim=1) 
        weights = self.weight_generator(combined)  
        
        weighted_base = base_sat * weights[:, 0:1]  
        weighted_residual = residual_sat * weights[:, 1:2]  

        fused = torch.cat([weighted_base, weighted_residual], dim=1) 
        output = self.fusion(fused)  
        output = self.output(output)  
        
        return output
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
       

class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, num_experts=2):
        super(DynamicConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.num_experts = num_experts
        
        self.expert_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
            for _ in range(num_experts)
        ])
        
        self.expert_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, num_experts, 1),
            nn.Softmax(dim=1)
        )
        
        self._init_weights()
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        expert_weights = self.expert_selector(x)  
        
        expert_outputs = []
        for i, expert_conv in enumerate(self.expert_convs):
            expert_out = expert_conv(x)  
            expert_outputs.append(expert_out)
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  
        expert_weights = expert_weights.unsqueeze(2) 
        
        output = torch.sum(expert_outputs * expert_weights, dim=1)  
        
        return output
    
    def _init_weights(self):
        for expert_conv in self.expert_convs:
            init.kaiming_uniform_(expert_conv.weight, a=0, mode='fan_in', nonlinearity='relu')
            if expert_conv.bias is not None:
                init.constant_(expert_conv.bias, 0)
        
        for m in self.expert_selector.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class ConditionalInstanceNorm(nn.Module):
    def __init__(self, num_features, condition_dim):
        super().__init__()
        self.num_features = num_features
        
        self.condition_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(condition_dim, num_features * 2, 1),
            nn.ReLU(),
            nn.Conv2d(num_features * 2, num_features * 2, 1)
        )
        
    def forward(self, x, condition):
        B, C, H, W = x.shape
        
        mean = x.view(B, C, -1).mean(dim=2, keepdim=True).view(B, C, 1, 1)
        var = x.view(B, C, -1).var(dim=2, keepdim=True).view(B, C, 1, 1)
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        
        params = self.condition_net(condition) 
        gamma, beta = torch.chunk(params, 2, dim=1) 
        out = gamma * x_norm + beta
        return out


class ContrastiveFusionModule(nn.Module):
    def __init__(self, channel_dim=64, embed_dim=32):
        super().__init__()
        self.channel_dim = channel_dim
        self.embed_dim = embed_dim
        
        self.h_proj = nn.Sequential(
            nn.Conv2d(channel_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU()
        )
        
        self.s_proj = nn.Sequential(
            nn.Conv2d(channel_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU()
        )
        
        self.v_proj = nn.Sequential(
            nn.Conv2d(channel_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU()
        )
        
        self.contrastive_net = nn.Sequential(
            nn.Conv2d(embed_dim * 3, embed_dim * 2, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim * 2, embed_dim, 1),
            nn.Sigmoid()
        )
        

        self.cin_h = ConditionalInstanceNorm(embed_dim, embed_dim * 2)
        self.cin_s = ConditionalInstanceNorm(embed_dim, embed_dim * 2)
        self.cin_v = ConditionalInstanceNorm(embed_dim, embed_dim * 2)
        
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim * 3, embed_dim, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, 3, 1),
            nn.Softmax(dim=1)
        )
        
        self.fusion_net = nn.Sequential(
            nn.Conv2d(embed_dim * 3, embed_dim * 2, 3, padding=1),
            nn.ReLU(),
            SEBlock(embed_dim * 2, reduction_ratio=4),
            nn.Conv2d(embed_dim * 2, embed_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU()
        )
        
        self.output_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim * 2, channel_dim, 1)
        )
        
        self._init_weights()
    
    def forward(self, h_feat, s_feat, v_feat):
        h_embed = self.h_proj(h_feat)  
        s_embed = self.s_proj(s_feat)  
        v_embed = self.v_proj(v_feat)  
        
        concat_embed = torch.cat([h_embed, s_embed, v_embed], dim=1)  
        contrastive_features = self.contrastive_net(concat_embed)  
        
        condition = torch.cat([contrastive_features, contrastive_features], dim=1)  

        h_normalized = self.cin_h(h_embed, condition)
        s_normalized = self.cin_s(s_embed, condition)
        v_normalized = self.cin_v(v_embed, condition)

        concat_normalized = torch.cat([h_normalized, s_normalized, v_normalized], dim=1)  
        fusion_weights = self.weight_net(concat_normalized)  

        weighted_h = h_normalized * fusion_weights[:, 0:1, :, :]
        weighted_s = s_normalized * fusion_weights[:, 1:2, :, :]
        weighted_v = v_normalized * fusion_weights[:, 2:3, :, :]
        
        weighted_concat = torch.cat([weighted_h, weighted_s, weighted_v], dim=1) 
        fused = self.fusion_net(weighted_concat) 
        
        output = self.output_proj(fused) 
        
        return output
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class StructuralLossComponents(nn.Module):
    def __init__(self):
        super().__init__()
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

        self.C3 = 0.015 ** 2  
        
    def compute_ssim_components(self, x, y):
        """计算SSIM的各个组件"""
        mu_x = torch.mean(x, dim=[2, 3], keepdim=True)
        mu_y = torch.mean(y, dim=[2, 3], keepdim=True)
        
        sigma_x = torch.var(x, dim=[2, 3], keepdim=True)
        sigma_y = torch.var(y, dim=[2, 3], keepdim=True)
        sigma_xy = torch.mean((x - mu_x) * (y - mu_y), dim=[2, 3], keepdim=True)
        
        l = (2 * mu_x * mu_y + self.C1) / (mu_x ** 2 + mu_y ** 2 + self.C1)
        c = (2 * torch.sqrt(sigma_x * sigma_y + 1e-6) + self.C2) / (sigma_x + sigma_y + self.C2)
        s = (sigma_xy + self.C3) / (torch.sqrt(sigma_x * sigma_y + 1e-6) + self.C3)
        
        return l, c, s
    
    def structural_consistency_loss(self, pred, target):
        l, c, s = self.compute_ssim_components(pred, target)
        ssim = l * c * s
        ssim_loss = 1 - torch.mean(ssim)
        
        local_ssim = self.compute_local_ssim(pred, target)
        local_loss = 1 - torch.mean(local_ssim)
        
        return ssim_loss + 0.3 * local_loss
    
    def compute_local_ssim(self, pred, target, window_size=11):
        mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
        mu_target = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu_pred_sq
        sigma_target = F.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size//2) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu_pred_target
        
        ssim_map = ((2 * mu_pred_target + self.C1) * (2 * sigma_pred_target + self.C2)) / \
                   ((mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred + sigma_target + self.C2))
        
        return ssim_map
    
    def edge_preservation_loss(self, pred, target):
        pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        
        sobel_x = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], dtype=torch.float32, device=pred.device)
        sobel_y = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], dtype=torch.float32, device=pred.device)

        pred_grad_x = F.conv2d(pred_gray, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_gray, sobel_y, padding=1)
        target_grad_x = F.conv2d(target_gray, sobel_x, padding=1)
        target_grad_y = F.conv2d(target_gray, sobel_y, padding=1)
        
        pred_grad_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-6)
        target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-6)
        
        edge_loss = F.mse_loss(pred_grad_mag, target_grad_mag)
        
        pred_grad_dir = torch.atan2(pred_grad_y, pred_grad_x + 1e-6)
        target_grad_dir = torch.atan2(target_grad_y, target_grad_x + 1e-6)
        dir_loss = F.mse_loss(torch.sin(pred_grad_dir), torch.sin(target_grad_dir))
        
        return edge_loss + 0.2 * dir_loss

class LYT(nn.Module):
    def __init__(self, denoiser=None, teacher_network=None, filters=64, use_gradient_guided=True):  
        super(LYT, self).__init__()
        self.teacher_network = teacher_network if teacher_network is not None else TeacherNetwork(
            embed_size=64, num_heads=8, feedforward_dim=256, use_gradient_guided=use_gradient_guided
        )
        
        self.circular_h_processor = CircularHProcessing(base_channels=32)
        self.process_h = self._create_processing_layers(filters)
        
        self.residual_quantization_s_processor = ResidualQuantizationSProcessor(base_channels=32)
        
        self.h_final = nn.Conv2d(filters, 1, kernel_size=3, padding=1)
        self.s_final = nn.Conv2d(64, 1, kernel_size=3, padding=1)  
        self.v_final = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        
        self.contrastive_fusion = ContrastiveFusionModule(channel_dim=64, embed_dim=32)
        
        self.h_channel_adapter = nn.Conv2d(filters, 64, kernel_size=1)
        
        self.feature_expansion = nn.Sequential(
            nn.Conv2d(64, 64, 1),  
            nn.ReLU(),
            SEBlock(64, reduction_ratio=4)  
        )
        self.feature_expansion_scale2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),  
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.feature_expansion_dilated = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, dilation=2), 
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),  
            nn.ReLU()
        )
        
        self.spatial_attention_fusion = nn.Sequential(
            nn.Conv2d(192, 192, 7, padding=3, groups=64),  
            nn.Sigmoid()
        )
        
        self.progressive_fusion_stage1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            SEBlock(64, reduction_ratio=4)
        )
        self.progressive_fusion_stage2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        
        
        self.final_adjustments = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            SEBlock(32, reduction_ratio=4),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()  
        )
        
        self.res_conv = nn.Conv2d(3, 64, 1)  
        
        self._init_weights()

    def _create_processing_layers(self, filters):
        return nn.Sequential(
            nn.Conv2d(1, filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def _rgb_to_hsv(self, image):
        r, g, b = image[:,0], image[:,1], image[:,2]
        maxc, _ = torch.max(image, dim=1)
        minc, _ = torch.min(image, dim=1)
        delta = maxc - minc
        delta_safe = delta + 1e-6

        v = maxc
        s = delta / (maxc + 1e-6)
        h = torch.zeros_like(maxc)

        idx_nonzero = delta > 1e-6
        
        if idx_nonzero.any():
            idx_r = (maxc == r) & idx_nonzero
            idx_g = (maxc == g) & idx_nonzero
            idx_b = (maxc == b) & idx_nonzero
            
            h[idx_r] = ((g - b)[idx_r] / delta_safe[idx_r]) % 6
            h[idx_g] = ((b - r)[idx_g] / delta_safe[idx_g] + 2)
            h[idx_b] = ((r - g)[idx_b] / delta_safe[idx_b] + 4)
        
        h = h / 6

        return torch.stack([h, s, v], dim=1)  

    def forward(self, inputs):
        orig_input = inputs
        hsv = self._rgb_to_hsv(inputs)   
        h, s, v = torch.split(hsv, 1, dim=1)  
        
        h_enhanced = self.circular_h_processor(h)  
        h_processed = self.process_h(h_enhanced)  

        s_processed = self.residual_quantization_s_processor(s)  
        
        v_enhanced = self.teacher_network(v)  
        
        h_adjusted = self.h_channel_adapter(h_processed)  
        
        fused_channels = self.contrastive_fusion(h_adjusted, s_processed, v_enhanced) 
        
        fused_channels = fused_channels + 0.3 * self.progressive_fusion_stage2(
            self.progressive_fusion_stage1(fused_channels)
        )
        

        fused_channels_orig = self.feature_expansion(fused_channels)  
        fused_channels_scale2 = self.feature_expansion_scale2(fused_channels)  
        fused_channels_dilated = self.feature_expansion_dilated(fused_channels)  
        fused_channels_expanded = torch.cat([fused_channels_orig, fused_channels_scale2, fused_channels_dilated], dim=1)  
        
        spatial_attn = self.spatial_attention_fusion(fused_channels_expanded)  
        fused_channels_attended = fused_channels_expanded * spatial_attn  
        
        fused_features = self.feature_fusion(fused_channels_attended)  
        
        res_feat = self.res_conv(orig_input)
        enhanced_features = fused_features + 0.3 * res_feat  

        output = self.final_adjustments(enhanced_features)
        
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"Warning: Output contains NaN or Inf values")
        return output
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')  
                if m.bias is not None:
                    init.constant_(m.bias, 0) 

        if hasattr(self, 'final_adjustments'):
            last_conv = None
            for module in self.final_adjustments:
                if isinstance(module, nn.Conv2d):
                    last_conv = module
            if last_conv is not None:
                init.xavier_uniform_(last_conv.weight)
                if last_conv.bias is not None:
                    init.constant_(last_conv.bias, 0)

