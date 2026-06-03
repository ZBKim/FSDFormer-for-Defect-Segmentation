class DynamicConvSegHead(nn.Module):
    def __init__(self, channel=64, num_queries=16, num_dynamic_layers=3):
        super().__init__()
        self.num_queries = num_queries
        self.num_dynamic_layers = num_dynamic_layers
        
        self.multi_scale_mask = MultiScaleMaskFeatures(channel)
        
        self.decoder_norm = nn.LayerNorm(channel)
        self.class_embed = nn.Linear(channel, 1)
        
        # 第一层: channel -> channel (3x3 conv)
        self.controller_layer1 = nn.Linear(
            channel, 
            channel * channel * 3 * 3  # 卷积核权重
        )
        self.controller_bias1 = nn.Linear(channel, channel)
        
        # 第二层: channel -> channel (3x3 conv)
        self.controller_layer2 = nn.Linear(
            channel,
            channel * channel * 3 * 3
        )
        self.controller_bias2 = nn.Linear(channel, channel)
        
        # 第三层: channel -> 1 (1x1 conv)
        self.controller_layer3 = nn.Linear(
            channel,
            channel * 1 * 1
        )
        self.controller_bias3 = nn.Linear(channel, 1)
        
        self.coord_conv = CoordConv(channel, channel)
        
        self.boundary_refine = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, output, d1, d2, d3, attn_mask_target_size=None):
        """
        output: [16, B, 64]
        """
        B = d1.size(0)
        N_q = output.size(0)
        

        mask_features = self.multi_scale_mask(d1, d2, d3)  # [B, 64, H, W]
        _, C, H, W = mask_features.shape

        mask_features = self.coord_conv(mask_features)  # [B, 64, H, W]

        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  # [B, 16, 64]
        
        outputs_class = self.class_embed(decoder_output)  # [B, 16, 1]
    

        weights1 = self.controller_layer1(decoder_output)  # [B, 16, C*C*9]
        weights1 = weights1.view(B, N_q, C, C, 3, 3)
        bias1 = self.controller_bias1(decoder_output)  # [B, 16, C]
        

        weights2 = self.controller_layer2(decoder_output)
        weights2 = weights2.view(B, N_q, C, C, 3, 3)
        bias2 = self.controller_bias2(decoder_output)
        

        weights3 = self.controller_layer3(decoder_output)
        weights3 = weights3.view(B, N_q, C, 1, 1, 1)
        bias3 = self.controller_bias3(decoder_output)  # [B, 16, 1]

        outputs_mask = []
        
        for i in range(B):
            feat = mask_features[i:i+1]  # [1, C, H, W]
            
            masks_per_image = []
            
            for q in range(N_q):

                x = F.conv2d(
                    feat,
                    weights1[i, q],  # [C, C, 3, 3]
                    bias=bias1[i, q],  # [C]
                    padding=1
                )  # [1, C, H, W]
                x = F.relu(x)
                
                x = F.conv2d(
                    x,
                    weights2[i, q],
                    bias=bias2[i, q],
                    padding=1
                )
                x = F.relu(x)
                
                x = F.conv2d(
                    x,
                    weights3[i, q].squeeze(-1).squeeze(-1),  # [C, 1, 1]
                    bias=bias3[i, q]
                )  # [1, 1, H, W]
                
                masks_per_image.append(x)
            
            masks_per_image = torch.cat(masks_per_image, dim=1)  # [1, 16, H, W]
            outputs_mask.append(masks_per_image)
        
        outputs_mask = torch.cat(outputs_mask, dim=0)  # [B, 16, H, W]
        
        refined_masks = []
        for i in range(N_q):
            mask_i = outputs_mask[:, i:i+1]  # [B, 1, H, W]
            refined = self.boundary_refine(torch.sigmoid(mask_i))
            refined_masks.append(refined)
        
        outputs_mask_refined = torch.cat(refined_masks, dim=1)  # [B, 16, H, W]
        outputs_mask = outputs_mask + outputs_mask_refined  # 残差
        
        return outputs_class, outputs_mask, None


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        xx = torch.linspace(-1, 1, W, device=x.device)
        yy = torch.linspace(-1, 1, H, device=x.device)
        yy, xx = torch.meshgrid(yy, xx, indexing='ij')
        
        xx = xx.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 1, H, W]
        yy = yy.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)

        x_with_coord = torch.cat([x, xx, yy], dim=1)  # [B, C+2, H, W]
        
        return self.conv(x_with_coord)