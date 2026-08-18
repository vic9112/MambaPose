import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from timm.layers.weight_init import trunc_normal_
import math
from easydict import EasyDict
import copy
import numpy as np
from mamba_ssm.modules.mamba_simple import Mamba

import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
from mmcv.cnn import build_conv_layer
from mmengine.dist import get_dist_info
from mmengine.structures import PixelData
from torch import Tensor, nn
from functools import partial

from .pif import PoseInteraction

MIN_NUM_PATCHES = 16
BN_MOMENTUM = 0.1


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
        d_model=256,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
        if_bimamba=False,
        bimamba_type="none",
        if_devide_out=False,
        init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, if_devide_out=if_devide_out,
                        init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        tmp_x, tok_attn, attn = self.fn(x, **kwargs)
        return tmp_x + x, tok_attn, attn
        # return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn, fusion_factor=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim * fusion_factor)
        self.fn = fn

    def forward(self, x, **kwargs):
        x, tok_attn, attn = self.fn(self.norm(x), **kwargs)
        return x, tok_attn, attn
        # return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x), None, None


class Attention(nn.Module):
    """
    Self-attention Module
    """

    def __init__(self, dim, heads=8, dropout=0., num_keypoints=None, scale_with_head=False):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5 if scale_with_head else dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.num_keypoints = num_keypoints

    def forward(self, x, mask=None, return_tok=False):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        if return_tok:
            # N = HW + J
            J = self.num_keypoints
            tok_attn = attn[:, :, :J, J:]  # (B, H, J, HW)
            tok_attn = tok_attn.sum(1) / self.heads  # (B, J, HW), average all head
            return out, tok_attn, attn
        else:
            return out, None, attn


class Transformer(nn.Module):
    """
    Vision-transformer Module
    """

    def __init__(self, dim, depth, heads, mlp_dim, dropout, num_keypoints=None, all_attn=False, scale_with_head=False,
                 pruning_loc=[3, 6, 9]):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.all_attn = all_attn
        self.num_keypoints = num_keypoints
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout, num_keypoints=num_keypoints,
                                                scale_with_head=scale_with_head))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))
        self.pruning_loc = pruning_loc

    def forward(self, x, mask=None, pos=None, prune=False, keep_ratio=0.7, pass_1_pos=None):
        if len(x.shape) == 2:
            _, C = x.shape
            B = 1
        else:
            B, _, C = x.shape

        pos = pos.expand(B, -1, -1)
        attn_res = []

        # use the positional embedding from pass 1 in training
        if pass_1_pos is not None:
            pos = pass_1_pos

        for idx, (attn, ff) in enumerate(self.layers):

            # >>>>>>>>>> add patch embedding >>>>>>>>>>
            if idx > 0 and self.all_attn:
                x[:, self.num_keypoints:] += pos  # adding embedding within transformer

            # >>>>>>>>>> Attention layer >>>>>>>>>>
            if idx in self.pruning_loc and prune and keep_ratio < 1:
                x, tok_attn, x_attn = attn(x, mask=mask, return_tok=True)
                joint_tok_copy = x[:, :self.num_keypoints]  # (B, J, C)     save token

                B, _, num_patches = tok_attn.shape  # num_patch = HW
                num_keep_node = math.ceil(num_patches * keep_ratio)  # K = HW * ratio

                # attentive token
                human_attn = tok_attn.sum(1)  # (B, HW)
                attentive_idx = human_attn.topk(num_keep_node, dim=1)[1]  # (B, K)        without gradient
                attentive_idx = attentive_idx.unsqueeze(-1).expand(-1, -1, C)  # (B, K, C)
                x_attentive = torch.gather(x[:, self.num_keypoints:], dim=1,
                                           index=attentive_idx)  # (B, N, C) -> (B, K, C)
                pos = torch.gather(pos, dim=1, index=attentive_idx)  # (B, N, C) -> (B, K, C)

                x = torch.cat([joint_tok_copy, x_attentive], dim=1)
                x, _, _ = ff(x)
                attn_res.append(x_attn)
            else:
                x, _, x_attn = attn(x, mask=mask)
                x, _, _ = ff(x)
                attn_res.append(x_attn)

            # if idx == 3:
            #     kpt_token = x[:, 0:self.num_keypoints]
            #     vis_token = x[:, self.num_keypoints:]
            #
            #     y = torch.cat((kpt_token, vis_token), dim=1)
            #     # y = vis_token  # mamba_tokenpose_coco_256x192_300ep_3Back_selfscan
            #     similarity = torch.matmul(kpt_token, y.transpose(1,
            #                                                      2))  # x 是 (batchsize, 18, 3136)，y 是 (batchsize, 100, 256)，需确保最后一维匹配
            #     topk_num = 3
            #     # 获取 topk 相似度的值和索引
            #     topk_values, topk_indices = torch.topk(similarity, k=topk_num, dim=-1)
            #     batch_indices = torch.arange(x.size(0)).view(-1, 1, 1)
            #     expanded_y = y[batch_indices, topk_indices]
            #     expanded_y = expanded_y.view(x.size(0), -1, y.size(2))
            #     result, residual = self.Mamba_selfScanBlock(torch.flip(expanded_y, [1]), None, inference_params=None)
            #     expanded_y_new = torch.flip(result, [1])[:, 0::topk_num, :]
            #     kpt_token_new = kpt_token + self.within_dropout_2(expanded_y_new)
            #     kpt_token_new = self.within_norm_2(kpt_token_new)
            #
            #     x = torch.cat((kpt_token_new, vis_token), dim=1)

        return x, attn_res, pos


class Transformer_sd(nn.Module):
    """
    Vision-transformer Module
    """

    def __init__(self, dim, depth, heads, mlp_dim, dropout, num_keypoints=None, all_attn=False, scale_with_head=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.all_attn = all_attn
        self.num_keypoints = num_keypoints
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout,
                                                scale_with_head=scale_with_head))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None, pos=None):
        res = []
        attn_res = []
        for idx, (attn, ff) in enumerate(self.layers):
            if idx > 0 and self.all_attn:
                x[:, self.num_keypoints:] += pos

            # x = attn(x, mask = mask)
            x, _, x_attn = attn(x, mask=mask)
            x, _, _ = ff(x)
            res.append(x)
            attn_res.append(x_attn)

        return res, attn_res




class TokenPose_TB_base(nn.Module):
    def __init__(self, *, feature_size, patch_size, num_keypoints, dim, depth, heads,
                 mlp_ratio, apply_init=False, apply_multi=True, heatmap_size=[64, 48],
                 patch_dim=0, dropout=0., emb_dropout=0.,
                 pos_embedding_type="learnable", pif_mode='full'):
        """
        TokenPose base head, heatmap-based prediction head.
        """
        super().__init__()
        assert isinstance(feature_size, list) and isinstance(patch_size, list), \
            'image_size and patch_size should be list'
        assert feature_size[0] % patch_size[0] == 0 and \
               feature_size[1] % patch_size[1] == 0, \
            'Image dimensions must be divisible by the patch size.'
        num_patches = (feature_size[0] // (patch_size[0])) * (feature_size[1] // (patch_size[1]))

        assert pos_embedding_type in ['sine', 'learnable', 'sine-full']

        self.inplanes = 64
        self.patch_dim = patch_dim
        self.patch_size = patch_size
        self.heatmap_size = heatmap_size
        hidden_heatmap_dim = heatmap_size[0] * heatmap_size[1] // 8
        heatmap_dim = heatmap_size[0] * heatmap_size[1]
        self.num_keypoints = num_keypoints
        self.num_patches = num_patches
        self.pos_embedding_type = pos_embedding_type
        self.all_attn = (self.pos_embedding_type == "sine-full")
        mlp_dim = dim * mlp_ratio

        self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_keypoints, dim))
        h, w = feature_size[0] // (self.patch_size[0]), feature_size[1] // (self.patch_size[1])
        self._make_position_embedding(w, h, dim, pos_embedding_type)

        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.pose_interaction = PoseInteraction(
            dim=dim,
            num_keypoints=num_keypoints,
            mode=pif_mode,
            block_factory=create_block)

        # transformer
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout,
                                       num_keypoints=num_keypoints, all_attn=self.all_attn,
                                       scale_with_head=True)

        self.to_keypoint_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_heatmap_dim),
            nn.LayerNorm(hidden_heatmap_dim),
            nn.Linear(hidden_heatmap_dim, heatmap_dim)
        ) if (dim <= hidden_heatmap_dim * 0.5 and apply_multi) else nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, heatmap_dim)
        )
        trunc_normal_(self.keypoint_token, std=.02)
        if apply_init:
            self.apply(self._init_weights)

    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        '''
        d_model: embedding size in transformer encoder
        '''
        assert pe_type in ['none', 'learnable', 'sine', 'sine-full']
        if pe_type == 'none':
            self.pos_embedding = None
            print("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h
                self.pe_w = w
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + self.num_keypoints, d_model))
                trunc_normal_(self.pos_embedding, std=.02)
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                print("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2 * math.pi):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, feature, mask=None):  # 主结构 12/10 最有效果
        # transformer
        b, _, _, _ = feature.shape
        # print(feature.shape)
        x = feature.permute(0, 2, 3, 1).reshape(b, -1, self.patch_dim)
        x = self.patch_to_embedding(x)

        b, n, _ = x.shape

        keypoint_tokens = repeat(self.keypoint_token, '() n d -> b n d', b=b)
        if self.pos_embedding_type in ["sine", "sine-full"]:
            # print(x.shape, self.pos_embedding.shape)
            x += self.pos_embedding[:, :n]
            x = torch.cat((keypoint_tokens, x), dim=1)
        else:
            x = torch.cat((keypoint_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + self.num_keypoints)]
        x = self.dropout(x)

        # print(x.shape,self.pos_embedding.shape)

        x, _, _ = self.transformer(x, mask, self.pos_embedding)

        kpt_token = x[:, 0:self.num_keypoints]
        vis_token = x[:, self.num_keypoints:]

        kpt_token = self.pose_interaction(kpt_token)

        x = self.to_keypoint_token(kpt_token)
        x = self.mlp_head(x)
        x = rearrange(x, 'b c (p1 p2) -> b c p1 p2', p1=self.heatmap_size[0], p2=self.heatmap_size[1])

        output = EasyDict(
            vis_token=vis_token,
            kpt_token=kpt_token,
            pred=x
        )

        return output

    # def forward(self, feature, mask=None):  # 不考虑先验
    #     # transformer
    #     b, _, _, _ = feature.shape
    #     # print(feature.shape)
    #     x = feature.permute(0, 2, 3, 1).reshape(b, -1, self.patch_dim)
    #     x = self.patch_to_embedding(x)
    #
    #     b, n, _ = x.shape
    #
    #     keypoint_tokens = repeat(self.keypoint_token, '() n d -> b n d', b=b)
    #     if self.pos_embedding_type in ["sine", "sine-full"]:
    #         # print(x.shape, self.pos_embedding.shape)
    #         x += self.pos_embedding[:, :n]
    #         x = torch.cat((keypoint_tokens, x), dim=1)
    #     else:
    #         x = torch.cat((keypoint_tokens, x), dim=1)
    #         x += self.pos_embedding[:, :(n + self.num_keypoints)]
    #     x = self.dropout(x)
    #
    #     # print(x.shape,self.pos_embedding.shape)
    #
    #     x, _, _ = self.transformer(x, mask, self.pos_embedding)
    #
    #     kpt_token = x[:, 0:self.num_keypoints]
    #     vis_token = x[:, self.num_keypoints:]
    #
    #     # y = torch.cat((kpt_token, vis_token), dim=1)
    #     y = kpt_token  # mamba_tokenpose_coco_256x192_300ep_3Back_selfscan
    #     similarity = torch.matmul(kpt_token, y.transpose(1, 2))  # y 是 (batchsize, 100, 256)
    #     topk_num = 5
    #     # 获取 topk 相似度的值和索引
    #     topk_values, topk_indices = torch.topk(similarity, k=topk_num, dim=-1)
    #     batch_indices = torch.arange(x.size(0)).view(-1, 1, 1)
    #     expanded_y = y[batch_indices, topk_indices]
    #     # print(expanded_y.shape)
    #     expanded_y = expanded_y.view(-1, topk_num, y.size(2))
    #     result, residual = self.Mamba_selfScanBlock(torch.flip(expanded_y, [1]), None, inference_params=None)
    #     result = result.view(x.size(0), -1, y.size(2))
    #     expanded_y_new = torch.flip(result, [1])[:, 0::topk_num, :]
    #     kpt_token = kpt_token + self.param * self.within_dropout_2(expanded_y_new)
    #     kpt_token = self.within_norm_2(kpt_token)
    #
    #     mean_tensor = kpt_token.mean(dim=1, keepdim=True)
    #     # 将平均张量与原始张量拼接
    #     kpt_token = torch.cat((mean_tensor, kpt_token), dim=1)
    #
    #     # if self.num_keypoints == 14:
    #     #     scann_indices = [0, 14, 13, 14, 0,
    #     #                      1, 3, 5, 3, 1, 0,
    #     #                      7, 9, 11, 9, 7, 0,
    #     #                      8, 10, 12, 10, 8, 0,
    #     #                      2, 4, 6, 4, 2, 0]
    #     #     return_indices = [28, 9, 23, 8, 24, 7, 25, 11, 17, 12, 18, 13, 19, 2, 3]
    #     # else:  # 新扫描路径
    #     #     scann_indices = [0, 7, 5, 3, 1, 2, 4, 6, 0,
    #     #                      6, 8, 10, 8, 6, 0,
    #     #                      12, 14, 16, 14, 12, 0,
    #     #                      13, 15, 17, 15, 13, 0,
    #     #                      7, 9, 11, 9, 7, 0]
    #     #     return_indices = [32, 4, 5, 3, 6, 2, 13, 27, 12, 28, 11, 29, 19, 21, 18, 22, 17, 23]
    #
    #     # hidden_states = kpt_token[:, scann_indices, :]
    #     hidden_states = kpt_token
    #     hidden_states_normal, residual = self.MambaBlock(hidden_states, None, inference_params=None)
    #     hidden_states_filp, residual = self.MambaBlock2(torch.flip(hidden_states, [1]), None, inference_params=None)
    #     hidden_states = hidden_states_normal + torch.flip(hidden_states_filp, [1])
    #     # hidden_states = hidden_states[:, return_indices, :]
    #     kpt_token = kpt_token + self.within_dropout(hidden_states)
    #     kpt_token = self.within_norm(kpt_token)
    #     kpt_token = kpt_token[:, 1:, :]
    #
    #     x = self.to_keypoint_token(kpt_token)
    #     x = self.mlp_head(x)
    #     x = rearrange(x, 'b c (p1 p2) -> b c p1 p2', p1=self.heatmap_size[0], p2=self.heatmap_size[1])
    #
    #     output = EasyDict(
    #         vis_token=vis_token,
    #         kpt_token=kpt_token,
    #         pred=x
    #     )
    #
    #     return output

    # def forward(self, feature, mask=None):  # 不考虑循环
    #     # transformer
    #     b, _, _, _ = feature.shape
    #     # print(feature.shape)
    #     x = feature.permute(0, 2, 3, 1).reshape(b, -1, self.patch_dim)
    #     x = self.patch_to_embedding(x)
    #
    #     b, n, _ = x.shape
    #
    #     keypoint_tokens = repeat(self.keypoint_token, '() n d -> b n d', b=b)
    #     if self.pos_embedding_type in ["sine", "sine-full"]:
    #         # print(x.shape, self.pos_embedding.shape)
    #         x += self.pos_embedding[:, :n]
    #         x = torch.cat((keypoint_tokens, x), dim=1)
    #     else:
    #         x = torch.cat((keypoint_tokens, x), dim=1)
    #         x += self.pos_embedding[:, :(n + self.num_keypoints)]
    #     x = self.dropout(x)
    #
    #     # print(x.shape,self.pos_embedding.shape)
    #
    #     x, _, _ = self.transformer(x, mask, self.pos_embedding)
    #
    #     kpt_token = x[:, 0:self.num_keypoints]
    #     vis_token = x[:, self.num_keypoints:]
    #
    #     # y = torch.cat((kpt_token, vis_token), dim=1)
    #     y = kpt_token  # mamba_tokenpose_coco_256x192_300ep_3Back_selfscan
    #     similarity = torch.matmul(kpt_token, y.transpose(1, 2))  # y 是 (batchsize, 100, 256)
    #     topk_num = 5
    #     # 获取 topk 相似度的值和索引
    #     topk_values, topk_indices = torch.topk(similarity, k=topk_num, dim=-1)
    #     batch_indices = torch.arange(x.size(0)).view(-1, 1, 1)
    #     expanded_y = y[batch_indices, topk_indices]
    #     # print(expanded_y.shape)
    #     expanded_y = expanded_y.view(-1, topk_num, y.size(2))
    #     result, residual = self.Mamba_selfScanBlock(torch.flip(expanded_y, [1]), None, inference_params=None)
    #     result = result.view(x.size(0), -1, y.size(2))
    #     expanded_y_new = torch.flip(result, [1])[:, 0::topk_num, :]
    #     kpt_token = kpt_token + self.param * self.within_dropout_2(expanded_y_new)
    #     kpt_token = self.within_norm_2(kpt_token)
    #
    #     mean_tensor = kpt_token.mean(dim=1, keepdim=True)
    #     # 将平均张量与原始张量拼接
    #     kpt_token = torch.cat((mean_tensor, kpt_token), dim=1)
    #
    #     if self.num_keypoints == 14: # 不考虑循环
    #         scann_indices = [0, 14, 13,
    #                          1, 3, 5,
    #                          7, 9, 11,
    #                          8, 10, 12,
    #                          2, 4, 6]
    #         return_indices = [0, 3,12,4,13,5, 14,6,9,7,10, 8,11,2,1 ]
    #     else:  # 新扫描路径
    #         scann_indices = [0, 7, 5, 3, 1, 2, 4, 6, 0,
    #                          6, 8, 10, 8, 6, 0,
    #                          12, 14, 16, 14, 12, 0,
    #                          13, 15, 17, 15, 13, 0,
    #                          7, 9, 11, 9, 7, 0]
    #         return_indices = [32, 4, 5, 3, 6, 2, 13, 27, 12, 28, 11, 29, 19, 21, 18, 22, 17, 23]
    #
    #     hidden_states = kpt_token[:, scann_indices, :]
    #     hidden_states_normal, residual = self.MambaBlock(hidden_states, None, inference_params=None)
    #     hidden_states_filp, residual = self.MambaBlock2(torch.flip(hidden_states, [1]), None, inference_params=None)
    #     hidden_states = hidden_states_normal + torch.flip(hidden_states_filp, [1])
    #     hidden_states = hidden_states[:, return_indices, :]
    #     kpt_token = kpt_token + self.within_dropout(hidden_states)
    #     kpt_token = self.within_norm(kpt_token)
    #     kpt_token = kpt_token[:, 1:, :]
    #
    #     x = self.to_keypoint_token(kpt_token)
    #     x = self.mlp_head(x)
    #     x = rearrange(x, 'b c (p1 p2) -> b c p1 p2', p1=self.heatmap_size[0], p2=self.heatmap_size[1])
    #
    #     output = EasyDict(
    #         vis_token=vis_token,
    #         kpt_token=kpt_token,
    #         pred=x
    #     )
    #
    #     return output
