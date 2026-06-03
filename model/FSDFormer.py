# @Zhengbei Jin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from model.pvtv2 import pvt_v2_b2,pvt_v2_b4

from model.CSSI import CrossScaleInteraction

from typing import Optional

from model.position_encoding import PositionEmbeddingSine
from model.transformer import Transformer, SelfAttentionLayer,  MLP, _get_activation_fn



class BoundaryHead(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, groups=channel),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, 1)
        )

    def forward(self, x):
        return self.head(x)


class WTSC_Attention_2D(nn.Module):
    def __init__(self, channels, reduction=4, eps=1e-6, learnable_wavelet=True,dropout=0.1):
        super().__init__()

        self.channels = channels
        self.eps = eps

        self.dwt_h = nn.Conv2d(
            channels, channels * 2,
            kernel_size=(2, 1), stride=(2, 1),
            groups=channels, bias=False
        )
        self.dwt_w = nn.Conv2d(
            channels * 2, channels * 4,
            kernel_size=(1, 2), stride=(1, 2),
            groups=channels * 2, bias=False
        )
        
        with torch.no_grad():

            w_h = torch.zeros(channels * 2, 1, 2, 1)
            for i in range(channels):
                w_h[i * 2, 0, :, 0] = torch.tensor([0.5, 0.5])      # 低通
                w_h[i * 2 + 1, 0, :, 0] = torch.tensor([0.5, -0.5])  # 高通
            self.dwt_h.weight.copy_(w_h)
            
            # 垂直方向: 每个group有2个output channels
            # weight shape: [channels*4, 1, 1, 2]
            w_w = torch.zeros(channels * 4, 1, 1, 2)
            for i in range(channels * 2):
                w_w[i * 2, 0, 0, :] = torch.tensor([0.5, 0.5])      # 低通
                w_w[i * 2 + 1, 0, 0, :] = torch.tensor([0.5, -0.5])  # 高通
            self.dwt_w.weight.copy_(w_w)
        
        self.dwt_h.weight.requires_grad = learnable_wavelet
        self.dwt_w.weight.requires_grad = learnable_wavelet

        # 处理高频细节
        self.sparse_attn = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channels * 3, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

        # 处理低频近似
        self.low_proj = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.BatchNorm2d(channels // reduction)
        )
        self.low_expand = nn.Conv2d(channels // reduction, channels, 1)

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )


        self.cross_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        self.recon = nn.Conv2d(channels, channels, 3, padding=1)
        self.dropout = nn.Dropout2d(dropout)
    def forward(self, x):
        B, C, H, W = x.shape
        identity = x
        coeffs_h = self.dwt_h(x)
        coeffs = self.dwt_w(coeffs_h)
        
        cLL, cLH, cHL, cHH = coeffs.chunk(4, dim=1)
        
        cHigh = torch.cat([cLH, cHL, cHH], dim=1)  # [B, 3C, H/2, W/2]


        sparse_weight = self.sparse_attn(cHigh)  # [B, C, 1, 1]

        cLL_centered = cLL - cLL.mean(dim=[2, 3], keepdim=True)
        

        cLL_flat = cLL_centered.flatten(2)  # [B, C, H*W/4]
        cov = torch.bmm(cLL_flat, cLL_flat.transpose(1, 2)) / (cLL_flat.shape[2] + self.eps)
        

        cov_mean = cov.mean(dim=2, keepdim=True).unsqueeze(-1)  # [B, C, 1, 1]
        low_weight = self.low_expand(self.low_proj(cov_mean))  # [B, C, 1, 1]
        low_weight = torch.sigmoid(low_weight)


        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_weight = self.spatial_attn(torch.cat([max_pool, avg_pool], dim=1))  # [B, 1, H, W]


        gate = self.cross_gate(torch.cat([sparse_weight, low_weight], dim=1))  # [B, C, 1, 1]
        
        # 组合所有注意力
        channel_attn = gate * sparse_weight * low_weight  # [B, C, 1, 1]
        

        enhanced = x * channel_attn * spatial_weight
        enhanced = self.dropout(enhanced) 
        # 重建
        output = self.recon(enhanced)

        return output + identity


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, 
            dropout=dropout,
            batch_first=False  # [L, B, C] format
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        
        tgt2, attn_weights = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, 
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2, _ = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, 
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x



class PQS(nn.Module):
    def __init__(self, channel, dropout=0.1):
        super(PQS, self).__init__()
        self.conv = convbnrelu(channel, channel, k=1, s=1, p=0)

        self.decoder_norm = nn.LayerNorm(channel)
        self.class_embed = nn.Linear(channel, 1)
        self.mask_embed = MLP(channel, channel, channel, 3)
        self.dropout = nn.Dropout(dropout)
    def forward(self, output, mask_features, attn_mask_target_size):

        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        decoder_output = self.dropout(decoder_output)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        mask_features = self.conv(mask_features)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        attn_mask = None

        return outputs_class, outputs_mask, attn_mask


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu),
        )

    def forward(self, x):
        return self.conv(x)


class FSDFormer(nn.Module):
    def __init__(self, method="pvt_v2_b2", channel=64, num_queries=16):
        super(FSDFormer, self).__init__()
        if method=="pvt_v2_b2":
            self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
            path = '/raid/jzb4/FSDFormer/pretrainModel/pvt_v2_b2.pth'
        else:
            self.backbone = pvt_v2_b4()  # [64, 128, 320, 512]
            path = '/raid/jzb4/FSDFormer/pretrainModel/pvt_v2_b4.pth'
            
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # Encoder translation layers
        self.Translayer1_1 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        self.latlayer1 = BasicConv2d(channel, channel, 1)
        self.latlayer2 = BasicConv2d(channel, channel, 1)
        self.latlayer3 = BasicConv2d(channel, channel, 1)
        self.latlayer4 = BasicConv2d(channel, channel, 1)

        self.outconv1 = convbnrelu(channel, channel, k=3, s=1, p=1)
        self.outconv2 = convbnrelu(channel, channel, k=3, s=1, p=1)
        self.outconv3 = convbnrelu(channel, channel, k=3, s=1, p=1)
        self.outconv4 = convbnrelu(channel, channel, k=3, s=1, p=1)


        self.csfi = CrossScaleInteraction(channels=channel, num_scales=4)

        self.mask_fusion = nn.Sequential(
            nn.Conv2d(channel * 4, channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )


        self.query_embed = nn.Embedding(num_queries, channel)

        # Boundary branch
        self.boundary_head = BoundaryHead(channel)
        
        self.wtsc_modules = nn.ModuleList()
        self.num_feature_levels = 3
        for _ in range(self.num_feature_levels):
            self.wtsc_modules.append(
                WTSC_Attention_2D(
                    channels=channel,
                    reduction=4,
                    learnable_wavelet=True
                )
            )
        
        # Positional encoding
        N_steps = channel // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.num_heads = 8
        
        # Transformer layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_feature_levels):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=channel,
                    nhead=self.num_heads,
                    dropout=0.1,
                    normalize_before=False,
                )
            )

            self.transformer_cross_attention_layers.append(
               CrossAttentionLayer(
                    d_model=channel,
                    nhead=self.num_heads,
                    dropout=0.1,
                    normalize_before=False,
                )
            )

        self.level_embed = nn.Embedding(self.num_feature_levels, channel)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            self.input_proj.append(convbnrelu(channel, channel, k=1, s=1, p=0))

        self.class_transformer = Transformer(
            d_model=channel,
            dropout=0.1,
            nhead=self.num_heads,
            dim_feedforward=2048,
            num_encoder_layers=0,
            num_decoder_layers=2,
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.class_input_proj = convbnrelu(channel, channel, k=1, s=1, p=0)
        self.PQSs = nn.ModuleList()
        for _ in range(self.num_feature_levels+1):
            self.PQSs.append(PQS(channel))

    def forward(self, x):
        image_shape = x.size()[2:]
        bs = x.size()[0]
        
        # Backbone
        pvt = self.backbone(x)

        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        x1_t = self.Translayer1_1(x1)
        x2_t = self.Translayer2_1(x2)
        x3_t = self.Translayer3_1(x3)
        x4_t = self.Translayer4_1(x4)
        
        # Cross-scale fusion
        features = [x1_t, x2_t, x3_t, x4_t]
        enhanced_features = self.csfi(features)
        d1, d2, d3, d4 = enhanced_features
        d1 = self.outconv1(d1)
        d2 = self.outconv2(d2)
        d3 = self.outconv3(d3)
        d4 = self.outconv4(d4)

        x = [d4, d3, d2]
        
        wtsc_enhanced_features = []
        for i in range(self.num_feature_levels):
            # 2D WTSC处理 [B, C, H, W] -> [B, C, H, W]
            enhanced = self.wtsc_modules[i](x[i])
            wtsc_enhanced_features.append(enhanced)
        
        src = []
        pos = []
        size_list = []
        for i in range(self.num_feature_levels):
            size_list.append(wtsc_enhanced_features[i].shape[-2:])
            
            pos.append(self.pe_layer(wtsc_enhanced_features[i], None).flatten(2))
            
            # [B, C, H, W] -> [B, C, HW] -> [HW, B, C]
            src.append(
                self.input_proj[i](wtsc_enhanced_features[i]).flatten(2) + 
                self.level_embed.weight[i][None, :, None]
            )
            
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)


        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        d2_up = F.interpolate(d2, size=d1.shape[-2:], mode='bilinear', align_corners=False)
        d3_up = F.interpolate(d3, size=d1.shape[-2:], mode='bilinear', align_corners=False)
        d4_up = F.interpolate(d4, size=d1.shape[-2:], mode='bilinear', align_corners=False)

        multi_scale = torch.cat([d1, d2_up, d3_up, d4_up], dim=1)

        mask_features = self.mask_fusion(multi_scale)

        boundary = torch.sigmoid(self.boundary_head(d1))
        mask_features = mask_features * (1.0 + boundary)

        out_t, _ = self.class_transformer(
            self.pe_layer(mask_features), None, 
            self.query_embed.weight,
            self.class_input_proj(mask_features), None
        )
        out_t = out_t[0]
        output = out_t
        predictions_mask = []

        outputs_class, outputs_mask, attn_mask = self.PQSs[0](
            output, mask_features, attn_mask_target_size=size_list[0]
        )
        predictions_mask.append(self.semantic_inference(outputs_class, outputs_mask))

        for i in range(self.num_feature_levels):
            level_index = i % self.num_feature_levels

            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],  # src已经是WTSC增强后的特征
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=pos[level_index], 
                query_pos=query_embed
            )
            
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            outputs_class, outputs_mask, attn_mask = self.PQSs[i+1](
                output, mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels]
            )
            predictions_mask.append(self.semantic_inference(outputs_class, outputs_mask))

        mask = predictions_mask[1] + predictions_mask[2] + predictions_mask[3]
        predictions_mask.append(mask)


        for i in range(self.num_feature_levels + 2):
            predictions_mask[i] = F.interpolate(
                predictions_mask[i], size=image_shape, mode='bilinear'
            )

        return predictions_mask

    def semantic_inference(self, mask_cls, mask_pred):
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        return semseg
    

