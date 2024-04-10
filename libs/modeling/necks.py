import torch
from torch import nn
from torch.nn import functional as F

from .models import register_neck
from .blocks import MaskedConv1D, LayerNorm,CustomTransformerEncoder,CustomTransformerEncoderLayer

@register_neck("fpn")
class FPN1D(nn.Module):
    """
        Feature pyramid network
    """
    def __init__(
        self,
        in_channels,      # input feature channels, len(in_channels) = # levels
        out_channel,      # output feature channel
        scale_factor=2.0, # downsampling rate between two fpn levels
        start_level=0,    # start fpn level
        end_level=-1,     # end fpn level
        with_ln=True      # if to apply layer norm at the end
    ):
        super().__init__()
        assert isinstance(in_channels, list) or isinstance(in_channels, tuple)

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # disable bias if using layer norm
            l_conv = MaskedConv1D(
                in_channels[i], out_channel, 1, bias=(not with_ln))
            # use depthwise conv here for efficiency
            fpn_conv = MaskedConv1D(
                out_channel, out_channel, 3,
                padding=1, bias=(not with_ln), groups=out_channel
            )
            # layer norm for order (B C T)
            if with_ln:
                fpn_norm = LayerNorm(out_channel)
            else:
                fpn_norm = nn.Identity()

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):
        # inputs must be a list / tuple
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) ==  len(self.in_channels)

        # build laterals, fpn_masks will remain the same with 1x1 convs
        laterals = []
        for i in range(len(self.lateral_convs)):
            x, _ = self.lateral_convs[i](
                inputs[i + self.start_level], fpn_masks[i + self.start_level]
            )
            laterals.append(x)

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i-1] += F.interpolate(
                laterals[i],
                scale_factor=self.scale_factor,
                mode='nearest'
            )

        # fpn conv / norm -> outputs
        # mask will remain the same
        fpn_feats = tuple()
        for i in range(used_backbone_levels):
            x, _ = self.fpn_convs[i](
                laterals[i], fpn_masks[i + self.start_level])
            x = self.fpn_norms[i](x)
            fpn_feats += (x, )

        return fpn_feats, fpn_masks

@register_neck('identity')
class FPNIdentity(nn.Module):
    def __init__(
        self,
        in_channels,      # input feature channels, len(in_channels) = # levels
        out_channel,      # output feature channel
        scale_factor=2.0, # downsampling rate between two fpn levels
        start_level=0,    # start fpn level
        end_level=-1,     # end fpn level
        with_ln=True      # if to apply layer norm at the end
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # check feat dims
            assert self.in_channels[i + self.start_level] == self.out_channel
            # layer norm for order (B C T)
            if with_ln:
                fpn_norm = LayerNorm(out_channel)
            else:
                fpn_norm = nn.Identity()
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):
        # inputs must be a list / tuple
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) ==  len(self.in_channels)

        # apply norms, fpn_masks will remain the same with 1x1 convs
        fpn_feats = tuple()
        for i in range(len(self.fpn_norms)):
            x = self.fpn_norms[i](inputs[i + self.start_level])
            fpn_feats += (x, )

        return fpn_feats, fpn_masks

class SimpleSelfAttention(nn.Module):
    def __init__(self, in_channel):
        super(SimpleSelfAttention, self).__init__()
        self.query = nn.Linear(in_channel, in_channel)
        self.key = nn.Linear(in_channel, in_channel)
        self.scale = in_channel ** -0.5

    def forward(self, x,mask):
        # x shape: [B, C, T] (Batch, Channel, Time/Length)
        B, C, T = x.shape
        # Global average pooling to get the feature vector
        x_pool = F.avg_pool1d(x, kernel_size=T).view(B, C)  # Shape: [B, C]
        # Query and Key
        query = self.query(x_pool)  # Shape: [B, C]
        key = self.key(x_pool)  # Shape: [B, C]
        # Attention score
        attention = F.softmax(query * key * self.scale, dim=-1)  # Shape: [B, C]
        # Apply attention
        x = x * attention.unsqueeze(-1)
        return x,mask
    
@register_neck('att_fpn')
class ATTfpn(nn.Module):
    """
    Feature pyramid network with self-attention for feature fusion.
    """
    def __init__(
        self,
        in_channels,
        out_channel,
        scale_factor=2.0,
        start_level=0,
        end_level=-1,
        with_ln=True
    ):
        super(ATTfpn, self).__init__()
        assert isinstance(in_channels, list) or isinstance(in_channels, tuple)

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        self.end_level = len(in_channels) if end_level == -1 else end_level
        assert self.end_level <= len(in_channels)
        assert self.start_level >= 0 and self.start_level < self.end_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.fpn_norms = nn.ModuleList()
        self.attention_modules = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            l_conv = MaskedConv1D(in_channels[i], out_channel, 1, bias=not with_ln)
            fpn_conv = MaskedConv1D(out_channel, out_channel, 3, padding=1, bias=not with_ln, groups=out_channel)
            attention_module = SimpleSelfAttention(out_channel)
            
            if with_ln:
                fpn_norm = LayerNorm(out_channel)  # Assuming T=1 after global pooling
            else:
                fpn_norm = nn.Identity()

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.attention_modules.append(attention_module)
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs,fpn_masks):
        assert len(fpn_masks) ==  len(self.in_channels)
        laterals = [self.lateral_convs[i](inputs[i + self.start_level],fpn_masks[i + self.start_level])[0] for i in range(len(self.lateral_convs))]

        # Apply self-attention before the top-down fusion
        for i, lateral in enumerate(laterals):       
            laterals[i],_ = self.attention_modules[i](lateral,fpn_masks[i+self.start_level])

        # Top-down path fusion
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=self.scale_factor, mode='nearest')

        # Final FPN convolution and normalization
        fpn_feats = [self.fpn_norms[i](self.fpn_convs[i](laterals[i],fpn_masks[i+self.start_level])[0]) for i in range(len(laterals))]

        return tuple(fpn_feats) ,fpn_masks

#### 使用特征层间的注意力机制

# class InterLayerAttention(nn.Module):
#     def __init__(self, num_layers, channel, feature_size):
#         super(InterLayerAttention, self).__init__()
#         self.num_layers = num_layers
#         self.channel = channel
#         self.feature_size = feature_size
#         # Assume all layers' features will be resized to the same dimension
#         self.resize_convs = nn.ModuleList([
#             nn.Conv1d(channel, channel, 1) for _ in range(num_layers)
#         ])
#         self.softmax = nn.Softmax(dim=0)  # Softmax across layers

#     def forward(self, layer_features):
#         # layer_features: List of tensors of shape [B, C, T] from different FPN layers
#         resized_features = []
#         for i, features in enumerate(layer_features):
#             # Resize feature to have the same feature_size for all layers
#             resized = F.interpolate(features, size=self.feature_size, mode='linear', align_corners=True)
#             resized = self.resize_convs[i](resized)
#             resized_features.append(resized)
        
#         # Stack and apply softmax
#         stacked_features = torch.stack(resized_features, dim=0)  # Shape: [num_layers, B, C, feature_size]
#         attention_weights = self.softmax(stacked_features)
        
#         # Apply attention weights
#         weighted_features = attention_weights * stacked_features
#         aggregated_features = torch.sum(weighted_features, dim=0)  # Sum across layers
        
#         return aggregated_features

class TransformerInterLayerAttention(nn.Module):
    def __init__(self, num_layers, channel, feature_size):
        super(TransformerInterLayerAttention, self).__init__()
        self.num_layers = num_layers
        self.channel = channel
        self.feature_size = feature_size
        self.conv = nn.ModuleList()
        for _ in range(6):
            self.conv.append(MaskedConv1D(512,1,1,bias=False))
        # Transformer Encoder Layer
        encoder_layer = CustomTransformerEncoderLayer(d_model=1024, nhead=1, dim_feedforward=feature_size)
        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.resize_convs = nn.ModuleList([
            nn.Conv1d(channel, channel, 1) for _ in range(num_layers)
        ])

    def forward(self, layer_features,fpn_masks):
        # layer_features: List of tensors of shape [B, C, T] from different FPN layers
        
        # Resize and prepare features for Transformer Encoder
        resized_features = []
        for i,features in enumerate(layer_features):
            features,_ = self.conv[i](features,fpn_masks[i])
            # Resize feature to have the same feature_size for all layers
            resized = F.interpolate(features, size=self.feature_size, mode='linear', align_corners=True)
            resized_features.append(resized)
        
        # Stack along the new dimension for transformer input: [S, N, E]
        # S: source sequence length, N: batch size, E: embedding size/channel
        transformer_input = torch.stack(resized_features, dim=0)
        print(f"input:{transformer_input.shape}")
        transformer_input = transformer_input.squeeze()
        transformer_input = transformer_input.permute(1,0,2)
        print(f"output:{transformer_input.shape}")

        # transformer_input = transformer_input.reshape()
        # transformer_input = transformer_input.permute(2, 1, 0, 3).flatten(2) # Reshape to [T, B, num_layers*channel]
        
        # Pass through Transformer Encoder
        transformer_output = self.transformer_encoder(transformer_input)
        print("fae")
        print(transformer_output.shape)
        # Optional: Reshape or process transformer_output here if necessary
        assert 1==2
        return transformer_output

@register_neck('merge_fpn')
class Merge_fpn(nn.Module):
    """
    Feature pyramid network with self-attention for feature fusion.
    """
    def __init__(
        self,
        in_channels,
        out_channel,
        scale_factor=2.0,
        start_level=0,
        end_level=-1,
        with_ln=True
    ):
        super(Merge_fpn, self).__init__()
        assert isinstance(in_channels, list) or isinstance(in_channels, tuple)

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        self.end_level = len(in_channels) if end_level == -1 else end_level
        assert self.end_level <= len(in_channels)
        assert self.start_level >= 0 and self.start_level < self.end_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.fpn_norms = nn.ModuleList()
        self.attention_modules = nn.ModuleList()
        self.attention_module = TransformerInterLayerAttention(self.end_level - self.start_level, out_channel, feature_size=1024)  # Example feature_size

        for i in range(self.start_level, self.end_level):
            l_conv = MaskedConv1D(in_channels[i], out_channel, 1, bias=not with_ln)
            fpn_conv = MaskedConv1D(out_channel, out_channel, 3, padding=1, bias=not with_ln, groups=out_channel)
            
            if with_ln:
                fpn_norm = LayerNorm(out_channel)  # Assuming T=1 after global pooling
            else:
                fpn_norm = nn.Identity()

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs,fpn_masks):
        assert len(fpn_masks) ==  len(self.in_channels)
        laterals = [self.lateral_convs[i](inputs[i + self.start_level],fpn_masks[i + self.start_level])[0] for i in range(len(self.lateral_convs))]

        # Final FPN convolution and normalization
        # fpn_feats = [self.fpn_norms[i](laterals[i]) for i in range(len(laterals))]
        fpn_feats = self.attention_module(laterals,fpn_masks)
        print(fpn_feats.shape)
        fpn_feats = fpn_feats.unsqueeze(0)
        fpn_mask = fpn_masks[0].unsqueeze(0)
        return tuple(fpn_feats) ,tuple(fpn_mask)