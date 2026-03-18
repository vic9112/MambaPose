# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence,Optional, Union
import copy
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import build_conv_layer, build_norm_layer, build_activation_layer
from mmengine.model import BaseModel
from easydict import EasyDict
from einops import rearrange, repeat
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
import torch.distributions as distributions
from ..transformer_heads.transformers.deformable_detr_layers import (DeformableDetrTransformerDecoder, DeformableDetrTransformerEncoder)
from torch.nn.init import normal_
from mmengine.model import xavier_init
from mmpose.registry import MODELS

def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def nlc_to_nchw(x, hw_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        hw_shape (Sequence[int]): The height and width of output feature map.

    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len does not match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W).contiguous()


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.

    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
    """
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()


class AdaptivePadding(nn.Module):
    """Applies padding to input (if needed) so that input can get fully covered
    by filter you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad zero around
    input. The "corner"  mode would pad zero to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel:
        stride (int | tuple): Stride of the filter. Default: 1:
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):

        super(AdaptivePadding, self).__init__()

        assert padding in ('same', 'corner')

        kernel_size = torch.nn.modules.utils.to_2tuple(kernel_size)
        stride = torch.nn.modules.utils.to_2tuple(stride)
        padding = torch.nn.modules.utils.to_2tuple(padding)
        dilation = torch.nn.modules.utils.to_2tuple(dilation)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [
                    pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2
                ])
        return x


class PatchEmbed(BaseModel):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The config dict for embedding
            conv layer type selection. Default: "Conv2d.
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: None (Would be set as `kernel_size`).
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only work when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        in_channels=3,
        embed_dims=768,
        conv_type='Conv2d',
        kernel_size=16,
        stride=16,
        padding='corner',
        dilation=1,
        bias=True,
        norm_cfg=None,
        input_size=None,
        init_cfg=None,
    ):
        super(PatchEmbed, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = torch.nn.modules.utils.to_2tuple(kernel_size)
        stride = torch.nn.modules.utils.to_2tuple(stride)
        dilation = torch.nn.modules.utils.to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adap_padding = None
        padding = torch.nn.modules.utils.to_2tuple(padding)

        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        if input_size:
            input_size = torch.nn.modules.utils.to_2tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adap_padding:
                pad_h, pad_w = self.adap_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (out_h, out_w).
        """

        if self.adap_padding:
            x = self.adap_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class PatchMerging(BaseModel):
    """Merge patch feature map.

    This layer groups feature map by kernel_size, and applies norm and linear
    layers to the grouped feature map. Our implementation uses `nn.Unfold` to
    merge patch, which is about 25% faster than original implementation.
    Instead, we need to modify pretrained models for compatibility.

    Args:
        in_channels (int): The num of input channels.
            to gets fully covered by filter and stride you specified..
            Default: True.
        out_channels (int): The num of output channels.
        kernel_size (int | tuple, optional): the kernel size in the unfold
            layer. Defaults to 2.
        stride (int | tuple, optional): the stride of the sliding blocks in the
            unfold layer. Default: None. (Would be set as `kernel_size`)
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int | tuple, optional): dilation parameter in the unfold
            layer. Default: 1.
        bias (bool, optional): Whether to add bias in linear layer or not.
            Defaults: False.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 stride=None,
                 padding='corner',
                 dilation=1,
                 bias=False,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if stride:
            stride = stride
        else:
            stride = kernel_size

        kernel_size = torch.nn.modules.utils.to_2tuple(kernel_size)
        stride = torch.nn.modules.utils.to_2tuple(stride)
        dilation = torch.nn.modules.utils.to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of unfold
            padding = 0
        else:
            self.adap_padding = None

        padding = torch.nn.modules.utils.to_2tuple(padding)
        self.sampler = nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride)

        sample_dim = kernel_size[0] * kernel_size[1] * in_channels

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, sample_dim)[1]
        else:
            self.norm = None

        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)

    def forward(self, x, input_size):
        """
        Args:
            x (Tensor): Has shape (B, H*W, C_in).
            input_size (tuple[int]): The spatial shape of x, arrange as (H, W).
                Default: None.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, Merged_H * Merged_W, C_out)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (Merged_H, Merged_W).
        """
        B, L, C = x.shape
        assert isinstance(input_size, Sequence), f'Expect ' \
                                                 f'input_size is ' \
                                                 f'`Sequence` ' \
                                                 f'but get {input_size}'

        H, W = input_size
        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W
        # Use nn.Unfold to merge patch. About 25% faster than original method,
        # but need to modify pretrained model for compatibility

        if self.adap_padding:
            x = self.adap_padding(x)
            H, W = x.shape[-2:]

        x = self.sampler(x)
        # if kernel_size=2 and stride=2, x should has shape (B, 4*C, H/2*W/2)

        out_h = (H + 2 * self.sampler.padding[0] - self.sampler.dilation[0] *
                 (self.sampler.kernel_size[0] - 1) -
                 1) // self.sampler.stride[0] + 1
        out_w = (W + 2 * self.sampler.padding[1] - self.sampler.dilation[1] *
                 (self.sampler.kernel_size[1] - 1) -
                 1) // self.sampler.stride[1] + 1

        output_size = (out_h, out_w)
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C
        x = self.norm(x) if self.norm else x
        x = self.reduction(x)
        return x, output_size


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def feature_sampling(mlvl_feats,
                     ref_pt,
                     pc_range,
                     img_metas,
                     no_sampling=False):
    """ sample multi-level features by projecting 3D reference points
            to 2D image
        Args:
            mlvl_feats (List[Tensor]): Image features from
                different level. Each element has shape
                (B, N, C, H_lvl, W_lvl).
            ref_pt (Tensor): The normalized 3D reference
                points with shape (bs, num_query, 3)
            pc_range: perception range of the detector
            img_metas (list[dict]): Meta information of multiple inputs
                in a batch, containing `lidar2img`.
            no_sampling (bool): If set 'True', the function will return
                2D projected points and mask only.
        Returns:
            ref_pt_3d (Tensor): A copy of original ref_pt
            sampled_feats (Tensor): sampled features with shape \
                (B C num_q N 1 fpn_lvl)
            mask (Tensor): Determine whether the reference point \
                has projected outsied of images, with shape \
                (B 1 num_q N 1 1)
    """
    lidar2img = [meta['lidar2img'] for meta in img_metas]
    lidar2img = np.asarray(lidar2img)
    lidar2img = ref_pt.new_tensor(lidar2img)
    ref_pt = ref_pt.clone()
    ref_pt_3d = ref_pt.clone()

    B, num_query = ref_pt.size()[:2]
    num_cam = lidar2img.size(1)
    eps = 1e-5

    ref_pt[..., 0:1] = \
        ref_pt[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]  # x
    ref_pt[..., 1:2] = \
        ref_pt[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]  # y
    ref_pt[..., 2:3] = \
        ref_pt[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]  # z

    # (B num_q 3) -> (B num_q 4) -> (B 1 num_q 4) -> (B num_cam num_q 4 1)
    ref_pt = torch.cat((ref_pt, torch.ones_like(ref_pt[..., :1])), -1)
    ref_pt = ref_pt.view(B, 1, num_query, 4)
    ref_pt = ref_pt.repeat(1, num_cam, 1, 1).unsqueeze(-1)
    # (B num_cam 4 4) -> (B num_cam num_q 4 4)
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4)\
                         .repeat(1, 1, num_query, 1, 1)
    # (... 4 4) * (... 4 1) -> (B num_cam num_q 4)
    pt_cam = torch.matmul(lidar2img, ref_pt).squeeze(-1)

    # (B num_cam num_q)
    z = pt_cam[..., 2:3]
    eps = eps * torch.ones_like(z)
    mask = (z > eps)
    pt_cam = pt_cam[..., 0:2] / torch.maximum(z, eps)  # prevent zero-division
    # padded nuscene image: 928*1600
    (h, w) = img_metas[0]['pad_shape']
    pt_cam[..., 0] /= w
    pt_cam[..., 1] /= h
    # else:
    # (h,w,_) = img_metas[0]['ori_shape'][0]          # waymo image
    # pt_cam[..., 0] /= w # cam0~2: 1280*1920
    # pt_cam[..., 1] /= h # cam3~4: 886 *1920 padded to 1280*1920
    # mask[:, 3:5, :] &= (pt_cam[:, 3:5, :, 1:2] < 0.7) # filter pt_cam_y > 886

    mask = (
        mask & (pt_cam[..., 0:1] > 0.0)
        & (pt_cam[..., 0:1] < 1.0)
        & (pt_cam[..., 1:2] > 0.0)
        & (pt_cam[..., 1:2] < 1.0))

    if no_sampling:
        return pt_cam, mask

    # (B num_cam num_q) -> (B 1 num_q num_cam 1 1)
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    mask = torch.nan_to_num(mask)

    pt_cam = (pt_cam - 0.5) * 2  # [0,1] to [-1,1] to do grid_sample
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B * N, C, H, W)
        pt_cam_lvl = pt_cam.view(B * N, num_query, 1, 2)
        sampled_feat = F.grid_sample(feat, pt_cam_lvl)
        # (B num_cam C num_query 1) -> List of (B C num_q num_cam 1)
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1)
        sampled_feat = sampled_feat.permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)

    sampled_feats = torch.stack(sampled_feats, -1)
    # (B C num_q num_cam fpn_lvl)
    sampled_feats = \
        sampled_feats.view(B, C, num_query, num_cam, 1, len(mlvl_feats))
    return ref_pt_3d, sampled_feats, mask


class Linear_with_norm(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super(Linear_with_norm, self).__init__()
        self.bias = bias
        self.norm = norm
        self.linear = nn.Linear(in_channel, out_channel, bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def forward(self, x):
        y = x.matmul(self.linear.weight.t())

        if self.norm:
            x_norm = torch.norm(x, dim=1, keepdim=True)
            y = y / x_norm

        if self.bias:
            y = y + self.linear.bias
        return y


@MODELS.register_module()
class PoseurTransformer(BaseModel):
    """ add noise training """

    def __init__(self,
                 as_two_stage=False,
                 num_feature_levels=4,
                 two_stage_num_proposals=300,
                 num_joints=17,
                 use_soft_argmax=False,
                 use_soft_argmax_def=False,
                 proposal_feature='backbone_s', # or encoder_memory
                 image_size=[192, 256],
                 init_q_sigmoid=False,
                 soft_arg_stride=4,
                 add_feat_2_query=False,
                 query_pose_emb=True,
                 num_noise_sample=3,
                 num_noise_verts=4,
                 noise_sigma=0.2,
                 embed_dims=256,
                 encoder_config = None,
                 decoder_config = None,
                 **kwargs):
        super(PoseurTransformer, self).__init__(**kwargs)
        assert query_pose_emb == True
        self.num_noise_sample = num_noise_sample
        self.num_noise_verts = num_noise_verts
        self.noise_sigma = noise_sigma
        self.add_feat_2_query = add_feat_2_query
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals
        try:
            self.embed_dims = self.encoder.embed_dims
        except:
            self.embed_dims = embed_dims
        self.num_joints = num_joints
        self.use_soft_argmax = use_soft_argmax
        self.use_soft_argmax_def = use_soft_argmax_def
        assert not (self.use_soft_argmax&self.use_soft_argmax_def)
        self.init_q_sigmoid = init_q_sigmoid
        self.image_size = image_size
        self.soft_arg_stride = soft_arg_stride
        self.proposal_feature = proposal_feature
        self.query_pose_emb = query_pose_emb
        self.prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2)*self.noise_sigma)
        self.encoder = DeformableDetrTransformerEncoder(**encoder_config)
        self.decoder = DeformableDetrTransformerDecoder(**decoder_config)
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_sigma = Linear_with_norm(self.embed_dims, self.num_joints * 2, norm=False)

        if self.use_soft_argmax:
            self.soft_argmax_coord = Heatmap1DHead(in_channels=self.embed_dims, expand_ratio=2, hidden_dims=(512, ), 
                                                    image_size=self.image_size, stride = self.soft_arg_stride)
            self.fc_layers = [self.fc_sigma]
        elif self.use_soft_argmax_def:
            self.soft_argmax_coord = Heatmap2DHead(in_channels=self.embed_dims,
                                                    image_size=self.image_size, stride = self.soft_arg_stride)
            self.fc_layers = [self.fc_sigma]
        else:
            self.fc_coord = Linear_with_norm(self.embed_dims, self.num_joints * 2)
            self.fc_layers = [self.fc_coord, self.fc_sigma]

        if self.query_pose_emb:
            self.pos_trans = nn.Linear(self.embed_dims * 2,
                                    self.embed_dims)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims)
            self.pos_embed = nn.Embedding(self.num_joints, self.embed_dims)
        else:
            self.pos_trans = nn.Linear(self.embed_dims * 2,
                                    self.embed_dims * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
        # else:
        #     self.reference_points = nn.Linear(self.embed_dims, 2)
        self.fp16_enabled = False

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, DeformableDetrTransformerEncoder):
                if hasattr(m, 'weight') and m.weight.dim() > 1:
                    xavier_init(m, distribution='uniform')
            if isinstance(m, DeformableDetrTransformerDecoder):
                if hasattr(m, 'weight') and m.weight.dim() > 1:
                    xavier_init(m, distribution='uniform')
        # if not self.as_two_stage:
        #     xavier_init(self.reference_points, distribution='uniform', bias=0.)
        normal_(self.level_embeds)
        if self.use_soft_argmax:
            self.soft_argmax_coord.init_weights()

        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask,
                                     spatial_shapes):
        """Generate proposals from encoded memory.
        Args:
            memory (Tensor) : The output of encoder,
                has shape (bs, num_key, embed_dim).  num_key is
                equal the number of points on feature map from
                all level.
            memory_padding_mask (Tensor): Padding mask for memory.
                has shape (bs, num_key).
            spatial_shapes (Tensor): The shape of all feature maps.
                has shape (num_level, 2).
        Returns:
            tuple: A tuple of feature map and bbox prediction.
                - output_memory (Tensor): The input of decoder,  \
                    has shape (bs, num_key, embed_dim).  num_key is \
                    equal the number of points on feature map from \
                    all levels.
                - output_proposals (Tensor): The normalized proposal \
                    after a inverse sigmoid, has shape \
                    (bs, num_keys, 4).
        """

        N, S, C = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H * W)].view(
                N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(
                    0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1),
                               valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            # proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            proposal = grid.view(N, -1, 2)
            proposals.append(proposal)
            _cur += (H * W)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) &
                                  (output_proposals < 0.99)).all(
                                      -1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.
        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_proposal_pos_embed(self,
                               proposals,
                               num_pos_feats=128,
                               temperature=10000):
        """Get the position embedding of proposal."""
        num_pos_feats = self.embed_dims // 2
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        # N, L, 2
        if self.init_q_sigmoid:
            proposals = proposals.sigmoid() * scale
        else:
            proposals = proposals * scale
        
        
        # N, L, 2, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 2, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def ref_po_gua(self, reference_points):
        # self.num_noise_sample
        # self.num_noise_verts
        DEVICE = reference_points.device
        if self.prior.loc.device != DEVICE:
            self.prior.loc = self.prior.loc.to(DEVICE)
            self.prior.scale_tril = self.prior.scale_tril.to(DEVICE)
            self.prior._unbroadcasted_scale_tril = self.prior._unbroadcasted_scale_tril.to(DEVICE)
            self.prior.covariance_matrix = self.prior.covariance_matrix.to(DEVICE)
            self.prior.precision_matrix = self.prior.precision_matrix.to(DEVICE)
        reference_points = reference_points.clone().detach()
        bs, k, _ = reference_points.shape
        reference_points = reference_points[:,None].repeat(1, self.num_noise_sample, 1, 1)

        offset_noise = self.prior.sample((bs, self.num_noise_sample, self.num_noise_verts))
        offset_noise = offset_noise.clip(-1, 1)

        rand_index = torch.randperm(self.num_noise_sample * self.num_joints, 
                            device=reference_points.device)[:self.num_noise_verts * self.num_noise_sample]
        rand_index = rand_index[None, :, None].repeat(bs, 1, 2)
        reference_points = rearrange(reference_points, 'b s k o -> b (s k) o')
        offset_noise = rearrange(offset_noise, 'b s k o -> b (s k) o')
        sampled_ref_point = torch.gather(reference_points, 1, rand_index)
        sampled_ref_point = sampled_ref_point + offset_noise
        sampled_ref_point = sampled_ref_point.clip(-0.7, 1.7)
        # reference_points_debug = reference_points.clone() #TODO remove this when unuse
        reference_points = reference_points.scatter_(1, rand_index, sampled_ref_point) 
        # reference_points = rearrange(reference_points, 'b (s k) o -> b s k o', 
        #                         s = self.num_noise_sample, k=17)

        return reference_points

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                reg_branches=None,
                fc_coord=None,
                cls_branches=None,
                **kwargs):
        
        assert self.as_two_stage or query_embed is not None

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)
        # [bs, H*W, num_lvls, 2]
        reference_points = \
            self.get_reference_points(spatial_shapes,
                                      valid_ratios,
                                      device=feat.device)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (H*W, bs, embed_dims)
        
        # memory = self.encoder(
        #     query=feat_flatten,
        #     key=None,
        #     value=None,
        #     query_pos=lvl_pos_embed_flatten,
        #     query_key_padding_mask=mask_flatten,
        #     spatial_shapes=spatial_shapes,
        #     reference_points=reference_points,
        #     level_start_index=level_start_index,
        #     valid_ratios=valid_ratios,
        #     **kwargs)

        memory = feat_flatten + lvl_pos_embed_flatten
        memory = memory.permute(1, 0, 2).contiguous()
        bs, _, c = memory.shape

        if self.proposal_feature=='backbone_l':
            x = mlvl_feats[0]
        elif self.proposal_feature=='backbone_s':
            x = mlvl_feats[-1]
            point_sample_feat = mlvl_feats[-1]
        elif self.proposal_feature=='encoder_memory_l':
            x = memory.permute(0, 2, 1)[:,:,:int(level_start_index[1])].view_as(mlvl_feats[0])
            point_sample_feat = memory.permute(0, 2, 1)[:,:,:int(level_start_index[1])].view_as(mlvl_feats[0])
        elif self.proposal_feature=='encoder_memory_s':
            x = memory.permute(0, 2, 1)[:,:,int(level_start_index[-1]):].view_as(mlvl_feats[-1])
        else:
            raise NotImplementedError

        BATCH_SIZE = x.shape[0]

        if self.use_soft_argmax:
            out_coord = self.soft_argmax_coord(x) # bs, 17, 2
            assert out_coord.shape[2] == 2
            x = self.avg_pool(x).reshape(BATCH_SIZE, -1)
            out_sigma = self.fc_sigma(x).reshape(BATCH_SIZE, self.num_joints, -1)
        elif self.use_soft_argmax_def:
            out_coord = self.soft_argmax_coord(x) # bs, 17, 2
            assert out_coord.shape[2] == 2
            x = self.avg_pool(x).reshape(BATCH_SIZE, -1)
            out_sigma = self.fc_sigma(x).reshape(BATCH_SIZE, self.num_joints, -1)
        else:
            x = self.avg_pool(x).reshape(BATCH_SIZE, -1)
            out_coord = self.fc_coord(x).reshape(BATCH_SIZE, self.num_joints, 2)
            assert out_coord.shape[2] == 2
            out_sigma = self.fc_sigma(x).reshape(BATCH_SIZE, self.num_joints, -1)

        # (B, N, 2)
        pred_jts = out_coord.reshape(BATCH_SIZE, self.num_joints, 2)
        sigma = out_sigma.reshape(BATCH_SIZE, self.num_joints, -1).sigmoid()
        scores = 1 - sigma
        # (B, N, 1)
        scores = torch.mean(scores, dim=2, keepdim=True)
        # pred_jts = pred_jts
        # reference_points = pred_jts.sigmoid()
        # reference_points = pred_jts.clip(0, 1)
        enc_outputs = EasyDict(
            pred_jts=pred_jts,
            sigma=sigma,
            maxvals=scores.float(),
        )

        # reference_points = (pred_jts.detach()).clip(0, 1)
        if self.training:
            reference_points = pred_jts.detach()
            reference_points_gus = self.ref_po_gua(reference_points)
            reference_points = torch.cat([reference_points, reference_points_gus], dim=1)
            reference_points_cliped = reference_points.clip(0, 1)
        else:
            reference_points = pred_jts.detach()
            reference_points_cliped = reference_points.clip(0, 1)

        init_reference_out = reference_points_cliped
        pred_jts_pos_embed = self.get_proposal_pos_embed(reference_points.detach())
        reference_points_pos_embed = self.get_proposal_pos_embed(reference_points_cliped.detach()) #query init here
        if self.add_feat_2_query:
            query_feat = point_sample(point_sample_feat, init_reference_out, align_corners=False).permute(0, 2, 1)
            reference_points_pos_embed = reference_points_pos_embed + query_feat
        query_pos_emb = torch.cat([pred_jts_pos_embed, reference_points_pos_embed], dim=2)
        pos_trans_out = self.pos_trans_norm(self.pos_trans(query_pos_emb))

        query = pos_trans_out

        if self.training:
            query_pos = self.pos_embed.weight.clone().repeat(bs, self.num_noise_sample + 1, 1).contiguous()
        else:
            query_pos = self.pos_embed.weight.clone().repeat(bs, 1, 1).contiguous()

        # decoder
        # query = query.permute(1, 0, 2)
        # memory = memory.permute(1, 0, 2)
        # query_pos = query_pos.permute(1, 0, 2)
        # print("query.shape", query.shape)
        # print("memory.shape", memory.shape)
        # print("query_pos.shape", query_pos.shape)
        # print("spatial_shapes",spatial_shapes)
        inter_states, inter_references = self.decoder(
            query=query.contiguous(),
            key=None,
            value=memory.contiguous(),
            query_pos=query_pos.contiguous(),
            key_padding_mask=mask_flatten.contiguous(),
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            fc_coord=fc_coord,
            **kwargs)

        inter_references_out = inter_references
        return memory.permute(1, 0, 2), spatial_shapes, level_start_index, inter_states, init_reference_out,\
            inter_references_out, enc_outputs
        # return memory, spatial_shapes, level_start_index, inter_states, init_reference_out,\
        #     inter_references_out, enc_outputs