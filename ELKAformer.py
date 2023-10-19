## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch

import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
# from Att import SEAttention
import torch.nn as nn
import thop
##################################################################################
##Channel_Attention
class CA(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        # 池化层，将每一个通道的宽和高都变为 1 (平均值)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            # 先降低维
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            # 再升维
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # y是权重
        return x * y.expand_as(x)

## Layer Norm
class CONV_LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

##########################################################################
## Bottleneck Block
class IBU(nn.Module):
    def __init__(self, dim):
        super(IBU, self).__init__()
        # self.norm = CONV_LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.conv1 = nn.Conv2d(dim, dim * 4, kernel_size=1, stride=1, padding=0)
        self.dwconv = nn.Conv2d(dim * 4, dim * 4, kernel_size=3, padding=1, stride=1, groups=dim * 4)
        self.conv2 = nn.Conv2d(dim * 4, dim, kernel_size=1, stride=1, padding=0)
        self.act = nn.Hardswish()

    def forward(self, x):
        # x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x1 = self.dwconv(x)
        x = x + x1
        x = self.act(x)
        x = self.conv2(x)
        return x
##########################################################################
## Detail Feature Extract Block
class ESRB(nn.Module):
    def __init__(self, dim=48):
        super(ESRB, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.bottle = IBU(dim//2)
        self.conv2 = nn.Conv2d(dim//2, dim//2, kernel_size=1)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x1, x2 = self.conv1(x).chunk(2, dim=1)
        n, c, h, w = x2.size()
        x1 = self.bottle(x1)  # (B,C/2,H,W)

        x2_h = self.bottle(self.pool_h(x2))  # (B,C/2,H,1)
        x2_w = self.bottle(self.pool_w(x2)).permute(0, 1, 3, 2)  # (B,C/2,1,W) --> (B,C/2,W,1)
        x2 = self.act(x2_h + x2_w)
        x2 = self.conv2(x1 * x2)

        out = torch.cat([x1, x2], dim=1)
        return self.conv(out)  # (B,C,H,W)

###############################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class DEFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super(DEFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv0 = nn.Conv2d(hidden_features, hidden_features, kernel_size=7, stride=1, padding=3,
                        groups=hidden_features, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=7, stride=1, padding=3,
                        groups=hidden_features, bias=bias)
        self.act1 = nn.GELU()
        self.act2 = nn.Sigmoid()

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x1, x2 = self.project_in(x).chunk(2, dim=1)
        x1 = self.act1(self.dwconv0(x1))
        x2 = self.act2(self.dwconv1(x2))
        out = x1 * x2
        return self.project_out(out)

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class LKAB(nn.Module):
    def __init__(self, dim, bias=False):
        super(LKAB, self).__init__()
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Hardswish(),
            nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )
        self.att = CA(dim, dim)
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        a = self.a(x)
        a = self.att(a)
        x = a * self.v(x)
        x = self.proj(x)
        return x

##########################################################################
class ELKB(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(ELKB, self).__init__()

        self.norm = CONV_LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.dfeb = ESRB(dim)
        self.norm1 = CONV_LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.attn = LKAB(dim, bias)
        self.norm2 = CONV_LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.ffn = DEFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x0 = x + self.dfeb(self.norm(x))
        x1 = x0 + self.attn(self.norm1(x0))
        out = x1 + self.ffn(self.norm2(x1))
        return out

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c, embed_dim, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Restormer -----------------------
class ELKAformer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,
                 num_blocks=[6, 4, 4, 6],
                 num_refinement_blocks=2,
                 # heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 # LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(ELKAformer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            ELKB(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias
                             ) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            ELKB(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            ELKB(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            ELKB(dim=int(dim * 2 ** 3), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            ELKB(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            ELKB(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            ELKB(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            ELKB(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1

###########################################################################################
model = ELKAformer().cuda()
x = torch.rand(1, 3, 32, 32).cuda()
print(model(x).size())


