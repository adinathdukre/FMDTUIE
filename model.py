"""Frequency Modulated Deformable Transformer for Underwater Image Enhancement (ICPR 2024).

Module names follow the released checkpoint. Mapping to the names used in the paper:

    MSFL        -> Multi-scale Feature Fusion Feed-forward module (MSF), with Adaptive Frequency Preservation (AFP)
    DFSL        -> Frequency Modulated Deformable Attention module (FMDA)
    SepConv2d   -> Spatio-Channel aware Modulated Deformable Convolution (SCMDC)
                   with the spatio-channel attentive offset extractor
    Fuse        -> Adaptive Edge Preserving skip connection (AEP)
    network     -> full encoder-decoder network
"""

import math
import numbers

import torch
import torch.nn as nn
from einops import rearrange
from torchvision.ops import DeformConv2d


class SwishActivation(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


def patch_fft_filter(x, weight, patch_size):
    """Split x into non-overlapping patches and scale each patch's spectrum by a learnable quantization matrix."""
    x_patch = rearrange(x, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=patch_size, p2=patch_size)
    x_fft = torch.fft.rfft2(x_patch.float()) * weight
    x_patch = torch.fft.irfft2(x_fft, s=(patch_size, patch_size))
    return rearrange(x_patch, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size)


##########################################################################
# Layer norm
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
# MSF: Multi-scale Feature Fusion Feed-forward module
class MSFL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.patch_size = 8
        self.dim = dim

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                   groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2,
                                   groups=hidden_features * 2, bias=bias)
        self.relu3 = SwishActivation()
        self.relu5 = SwishActivation()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1,
                                     groups=hidden_features, bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2,
                                     groups=hidden_features, bias=bias)
        self.relu3_1 = SwishActivation()
        self.relu5_1 = SwishActivation()

        # AFP: learnable quantization matrix applied to the patch-wise FFT
        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = patch_fft_filter(x, self.fft, self.patch_size)

        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)
        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_3, x2_5], dim=1)
        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))
        x = torch.cat([x1, x2], dim=1)
        return self.project_out(x)


##########################################################################
# Spatio-channel attentive offset extractor
class ChannelAttention(nn.Module):
    def __init__(self, in_planes=216, ratio=9):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.ca_fc1 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.ca_relu1 = nn.ReLU(inplace=True)
        self.ca_fc2 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid_channel = nn.Sigmoid()

    def forward(self, x):
        avg = self.ca_fc2(self.ca_relu1(self.ca_fc1(self.avg_pool(x))))
        mx = self.ca_fc2(self.ca_relu1(self.ca_fc1(self.max_pool(x))))
        return self.sigmoid_channel(avg + mx) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.conv1(torch.cat([avg_out, max_out], dim=1))
        return self.sigmoid(attn) * x


##########################################################################
# SCMDC: Spatio-Channel aware Modulated Deformable Convolution
class SepConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, act_layer=nn.ReLU):
        super().__init__()
        # `depthwise` is not used in forward(); it is kept so that the released checkpoint loads with strict=True.
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding, dilation=dilation, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # 216 = 3 x (8 groups x 3 x 3): x/y offsets and modulation mask for a grouped 3x3 deformable conv
        self.offset_conv1 = nn.Conv2d(in_channels, 216, 3, stride=1, padding=1, bias=False)
        self.deform1 = DeformConv2d(in_channels, out_channels, 3, padding=1, groups=8)
        self.CA = ChannelAttention()
        self.SA = SpatialAttention()

    def offset_gen(self, x):
        feat = self.SA(self.CA(x))
        o1, o2, mask = torch.chunk(feat, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        return offset, torch.sigmoid(mask)

    def forward(self, x):
        offset, mask = self.offset_gen(self.offset_conv1(x))
        x = self.act_layer(self.deform1(x, offset, mask))
        return self.pointwise(x)


##########################################################################
# FMDA: Frequency Modulated Deformable Attention
class DFSL(nn.Module):
    def __init__(self, dim, bias):
        super().__init__()
        kernel_size = 3
        stride = 1
        pad = (kernel_size - stride) // 2
        # NOTE: `bias` is passed positionally as `dilation` (=False -> 0) for SepConv2d. It only affects the
        # unused `depthwise` layer and is kept for checkpoint compatibility.
        self.to_q = SepConv2d(dim, dim, kernel_size, stride, pad, bias)
        self.to_k = SepConv2d(dim, dim, kernel_size, stride, pad, bias)
        self.to_v = SepConv2d(dim, dim, kernel_size, stride, pad, bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.norm = LayerNorm(dim, LayerNorm_type='WithBias')
        self.patch_size = 8

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        p = self.patch_size
        q_fft = torch.fft.rfft2(rearrange(q, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=p, p2=p).float())
        k_fft = torch.fft.rfft2(rearrange(k, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=p, p2=p).float())

        attn = torch.fft.irfft2(q_fft * k_fft, s=(p, p))
        attn = rearrange(attn, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=p, p2=p)
        attn = self.norm(attn)
        return self.project_out(v * attn)


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', att=False):
        super().__init__()
        self.att = att
        if self.att:
            self.norm1 = LayerNorm(dim, LayerNorm_type)
            self.attn = DFSL(dim, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = MSFL(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        if self.att:
            x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


##########################################################################
# AEP: Adaptive Edge Preserving skip connection
class downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1))
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        # The transpose/view pair is part of the trained model and is kept as is.
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        return self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B, H*W, C


class upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.deconv = nn.Sequential(nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2))
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))  # assumes square feature maps
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        return self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B, H*W, C


class Fuse(nn.Module):
    def __init__(self, n_feat, fuse=False):
        super().__init__()
        self.n_feat = n_feat
        self.patch_size = 8
        self.att_channel = TransformerBlock(dim=n_feat * 2)

        self.conv = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)
        self.dowsample_0 = downsample(in_channel=n_feat, out_channel=n_feat)
        self.upsample_poolup1 = upsample(in_channel=n_feat, out_channel=n_feat)
        self.dowsample_1 = downsample(in_channel=n_feat, out_channel=n_feat)
        self.upsample_poolup2 = upsample(in_channel=n_feat, out_channel=n_feat)
        self.fuse = fuse
        self.fft = nn.Parameter(torch.ones((n_feat, 1, 1, self.patch_size, self.patch_size // 2 + 1)))

    def forward(self, enc, dnc):
        enc = patch_fft_filter(enc, self.fft, self.patch_size)

        if self.fuse:
            low = self.upsample_poolup1(self.dowsample_0(enc))
        else:
            low = self.upsample_poolup2(self.dowsample_1(enc))
        B, L, C = low.shape
        H = W = int(math.sqrt(L))
        low = low.transpose(1, 2).contiguous().view(B, C, H, W)

        enc = enc - low  # keep high-frequency (edge) content
        x = self.conv(torch.cat((enc, dnc), dim=1))
        x = self.conv2(self.att_channel(x))
        e, d = torch.split(x, [self.n_feat, self.n_feat], dim=1)
        return e + d


##########################################################################
class OverlapPatchEmbed(nn.Module):
    """Overlapped image patch embedding with a 3x3 conv."""

    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat * 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat // 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)


##########################################################################
class network(nn.Module):
    """Frequency Modulated Deformable Transformer (3-level encoder-decoder).

    Input height and width must be equal and divisible by 32.
    """

    def __init__(self, inp_channels=3, out_channels=3, dim=8, num_blocks=(1, 1, 2), num_refinement_blocks=4,
                 ffn_expansion_factor=3, bias=False):
        super().__init__()
        num_blocks = [int(n) for n in num_blocks]

        def blocks(d, n, att):
            return nn.Sequential(*[TransformerBlock(dim=d, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                    att=att) for _ in range(n)])

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = blocks(dim, num_blocks[0], att=False)
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = blocks(dim * 2, num_blocks[1], att=False)
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = blocks(dim * 4, num_blocks[2], att=False)

        self.decoder_level3 = blocks(dim * 4, num_blocks[2], att=True)
        self.up3_2 = Upsample(dim * 4)
        # Defined in the released model but not used in forward(); kept for checkpoint compatibility.
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = blocks(dim * 2, num_blocks[1], att=True)
        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = blocks(dim, num_blocks[0], att=True)

        self.refinement = blocks(dim, num_refinement_blocks, att=True)

        self.fuse2 = Fuse(dim * 2)
        self.fuse1 = Fuse(dim, fuse=True)
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        out_enc_level1 = self.encoder_level1(self.patch_embed(inp_img))
        out_enc_level2 = self.encoder_level2(self.down1_2(out_enc_level1))
        out_enc_level3 = self.encoder_level3(self.down2_3(out_enc_level2))

        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.fuse2(self.up3_2(out_dec_level3), out_enc_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.fuse1(self.up2_1(out_dec_level2), out_enc_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        return self.output(out_dec_level1) + inp_img
