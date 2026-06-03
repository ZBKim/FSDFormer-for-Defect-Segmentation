import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFPN_Fusion(nn.Module):
    def __init__(self, channels, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.relu = nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
    def forward(self, x_high, x_low):
        _, _, H, W = x_low.size()
        x_high_up = F.interpolate(x_high, size=(H, W), 
                                 mode='bilinear', align_corners=False)
        
        w = self.relu(self.w1)
        w = w / (w.sum() + self.epsilon)
        out = w[0] * x_high_up + w[1] * x_low
        return self.conv(out)



class CrossScaleInteraction(nn.Module):
    def __init__(self, channels, num_scales=4):
        super().__init__()
        self.num_scales = num_scales
        self.scale_attn = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.Sigmoid()
            ) for _ in range(num_scales - 1)
        ])
        self.top_down_gates = nn.ModuleList([
            nn.Conv2d(channels * 2, channels, 1) 
            for _ in range(num_scales - 1)
        ])
        self.bottom_up_gates = nn.ModuleList([
            nn.Conv2d(channels * 2, channels, 1)
            for _ in range(num_scales - 1)
        ])
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        
    def forward(self, features):
        # Top-down pathway
        top_down = [features[-1]]
        for i in range(len(features) - 2, -1, -1):
            high_feat = F.interpolate(top_down[-1], size=features[i].shape[2:], 
                                     mode='bilinear', align_corners=False)
            
            concat = torch.cat([high_feat, features[i]], dim=1)
            gate = torch.sigmoid(self.top_down_gates[i](concat))
            fused = gate * high_feat + (1 - gate) * features[i]
            
            attn = self.scale_attn[i](concat)
            enhanced = fused * attn + features[i]
            
            top_down.append(enhanced)
        
        top_down = top_down[::-1]
        
        # Bottom-up pathway
        bottom_up = [top_down[0]]
        for i in range(1, len(features)):
            low_feat = F.max_pool2d(bottom_up[-1], kernel_size=2)
            
            concat = torch.cat([low_feat, top_down[i]], dim=1)
            gate = torch.sigmoid(self.bottom_up_gates[i-1](concat))
            fused = gate * low_feat + (1 - gate) * top_down[i]
            
            bottom_up.append(fused)
        
        weights = F.softmax(self.scale_weights, dim=0)
        output = []
        for i in range(len(features)):
            output.append(weights[i] * bottom_up[i])
        
        return output


