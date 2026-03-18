# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
from mmcv.cnn import build_conv_layer
from mmengine.dist import get_dist_info
from mmengine.structures import PixelData
from torch import Tensor, nn
from functools import partial

from mmpose.codecs.utils import get_simcc_normalized
from mmpose.evaluation.functional import simcc_pck_accuracy
from mmpose.models.utils.tta import flip_vectors
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptSampleList)
from ..base_head import BaseHead

from mamba_ssm.modules.mamba_simple import Mamba

OptIntSeq = Optional[Sequence[int]]

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
        d_model=3136,
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



@MODELS.register_module()
class SimbaHead(BaseHead):

    _version = 2

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        input_size: Tuple[int, int],
        in_featuremap_size: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
        deconv_type: str = 'heatmap',
        deconv_out_channels: OptIntSeq = (256, 256, 256),
        deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
        deconv_num_groups: OptIntSeq = (16, 16, 16),
        conv_out_channels: OptIntSeq = None,
        conv_kernel_sizes: OptIntSeq = None,
        final_layer: dict = dict(kernel_size=1),
        loss: ConfigType = dict(type='KLDiscretLoss', use_target_weight=True),
        decoder: OptConfigType = None,
        init_cfg: OptConfigType = None,
    ):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        if deconv_type not in {'heatmap', 'vipnas'}:
            raise ValueError(
                f'{self.__class__.__name__} got invalid `deconv_type` value'
                f'{deconv_type}. Should be one of '
                '{"heatmap", "vipnas"}')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio
        self.loss_module = MODELS.build(loss)
        
        self.within_dropout = nn.Dropout(0.1)
        self.within_norm = nn.LayerNorm(3136)
        self.within_dropout_2 = nn.Dropout(0.1)
        self.within_norm_2 = nn.LayerNorm(3136)
        self.MambaBlock = create_block()
        self.Mamba_selfScanBlock = create_block()
        
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        num_deconv = len(deconv_out_channels) if deconv_out_channels else 0
        if num_deconv != 0:
            self.heatmap_size = tuple(
                [s * (2**num_deconv) for s in in_featuremap_size])

            # deconv layers + 1x1 conv
            self.deconv_head = self._make_deconv_head(
                in_channels=in_channels,
                out_channels=out_channels,
                deconv_type=deconv_type,
                deconv_out_channels=deconv_out_channels,
                deconv_kernel_sizes=deconv_kernel_sizes,
                deconv_num_groups=deconv_num_groups,
                conv_out_channels=conv_out_channels,
                conv_kernel_sizes=conv_kernel_sizes,
                final_layer=final_layer)

            if final_layer is not None:
                in_channels = out_channels
            else:
                in_channels = deconv_out_channels[-1]

        else:
            self.deconv_head = None

            if final_layer is not None:
                cfg = dict(
                    type='Conv2d',
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1)
                cfg.update(final_layer)
                self.final_layer = build_conv_layer(cfg)
            else:
                self.final_layer = None

            self.heatmap_size = in_featuremap_size

        # Define SimCC layers
        flatten_dims = self.heatmap_size[0] * self.heatmap_size[1]

        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)

        self.mlp_head_x = nn.Linear(flatten_dims, W)
        self.mlp_head_y = nn.Linear(flatten_dims, H)

    def _make_deconv_head(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        deconv_type: str = 'heatmap',
        deconv_out_channels: OptIntSeq = (256, 256, 256),
        deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
        deconv_num_groups: OptIntSeq = (16, 16, 16),
        conv_out_channels: OptIntSeq = None,
        conv_kernel_sizes: OptIntSeq = None,
        final_layer: dict = dict(kernel_size=1)
    ) -> nn.Module:
        """Create deconvolutional layers by given parameters."""

        if deconv_type == 'heatmap':
            deconv_head = MODELS.build(
                dict(
                    type='HeatmapHead',
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    deconv_out_channels=deconv_out_channels,
                    deconv_kernel_sizes=deconv_kernel_sizes,
                    conv_out_channels=conv_out_channels,
                    conv_kernel_sizes=conv_kernel_sizes,
                    final_layer=final_layer))
        else:
            deconv_head = MODELS.build(
                dict(
                    type='ViPNASHead',
                    in_channels=in_channels,
                    out_channels=out_channels,
                    deconv_out_channels=deconv_out_channels,
                    deconv_num_groups=deconv_num_groups,
                    conv_out_channels=conv_out_channels,
                    conv_kernel_sizes=conv_kernel_sizes,
                    final_layer=final_layer))

        return deconv_head

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:

        # print(feats[-1].shape)
        if self.deconv_head is None:
            feats = feats[-1]
            if self.final_layer is not None:
                feats = self.final_layer(feats)
        else:
            feats = self.deconv_head(feats)

        # flatten the output heatmap
        x = torch.flatten(feats, 2)
        # print(x.shape, "old")

        # 在第二维求平均，保持维度
        mean_tensor = x.mean(dim=1, keepdim=True)  # 结果形状为 [32, 1, 3136]
        # 将平均张量与原始张量拼接
        x = torch.cat((mean_tensor, x), dim=1)  # 结果形状为 [32, 18, 3136]
        scann_indices = [0, 7, 5, 3, 1, 2, 4, 6, 0,
                         6, 8, 10, 8, 6, 0,
                         7, 9, 11, 9, 7, 0,
                         12, 14, 16, 14, 12, 0,
                         13, 15, 17, 15, 13, 0, ]
        return_indices = [32, 4, 5, 3, 6, 2, 7, 1, 12, 18, 11, 17, 25, 31, 24, 30, 23, 29]

        hidden_states = x[:, scann_indices, :]
        hidden_states_normal, residual = self.MambaBlock(hidden_states, None, inference_params=None)
        hidden_states_filp, residual = self.MambaBlock(torch.flip(hidden_states, [1]), None, inference_params=None)
        hidden_states = hidden_states_normal + torch.flip(hidden_states_filp, [1])
        hidden_states = hidden_states[:, return_indices, :]
        x = x + self.within_dropout(hidden_states)
        x = self.within_norm(x)

        # x(32,18,3136)
        similarity = torch.matmul(x, x.transpose(1, 2))
        topk_num = 5
        topk_values, topk_indices = torch.topk(similarity, k=topk_num, dim=-1)
        # 使用高级索引，构造包含最相似向量的张量
        # topk_indices 的形状为 [32, 18, 5]，用它来索引 x
        batch_indices = torch.arange(x.size(0)).view(-1, 1, 1)  # [32, 1, 1]
        expanded_x = x[batch_indices, topk_indices]  # 得到 [32, 18, 5, 3136]
        # 将第3维和第2维合并，得到形状为 [32, 90, 3136]
        expanded_x = expanded_x.view(x.size(0), -1, x.size(2))
        # print(expanded_x.shape,"222")  # 输出: torch.Size([32, 90, 3136])
        result, residual = self.Mamba_selfScanBlock(torch.flip(expanded_x, [1]), None,inference_params=None)
        x_new = torch.flip(result, [1])[:, 0::topk_num, :]
        x = x + self.within_dropout_2(x_new)
        x = self.within_norm_2(x)

        x = x[:,1:,:]
        # print(x.shape, "renew")

        pred_x = self.mlp_head_x(x)
        pred_y = self.mlp_head_y(x)

        return pred_x, pred_y

    def predict(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        test_cfg: OptConfigType = {},
    ) -> InstanceList:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            List[InstanceData]: The pose predictions, each contains
            the following fields:

                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)
                - keypoint_x_labels (np.ndarray, optional): The predicted 1-D
                    intensity distribution in the x direction
                - keypoint_y_labels (np.ndarray, optional): The predicted 1-D
                    intensity distribution in the y direction
        """

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats

            _batch_pred_x, _batch_pred_y = self.forward(_feats)

            _batch_pred_x_flip, _batch_pred_y_flip = self.forward(_feats_flip)
            _batch_pred_x_flip, _batch_pred_y_flip = flip_vectors(
                _batch_pred_x_flip,
                _batch_pred_y_flip,
                flip_indices=flip_indices)

            batch_pred_x = (_batch_pred_x + _batch_pred_x_flip) * 0.5
            batch_pred_y = (_batch_pred_y + _batch_pred_y_flip) * 0.5
        else:
            batch_pred_x, batch_pred_y = self.forward(feats)

        preds = self.decode((batch_pred_x, batch_pred_y))

        if test_cfg.get('output_heatmaps', False):
            rank, _ = get_dist_info()
            if rank == 0:
                warnings.warn('The predicted simcc values are normalized for '
                              'visualization. This may cause discrepancy '
                              'between the keypoint scores and the 1D heatmaps'
                              '.')

            # normalize the predicted 1d distribution
            sigma = self.decoder.sigma
            batch_pred_x = get_simcc_normalized(batch_pred_x, sigma[0])
            batch_pred_y = get_simcc_normalized(batch_pred_y, sigma[1])

            B, K, _ = batch_pred_x.shape
            # B, K, Wx -> B, K, Wx, 1
            x = batch_pred_x.reshape(B, K, 1, -1)
            # B, K, Wy -> B, K, 1, Wy
            y = batch_pred_y.reshape(B, K, -1, 1)
            # B, K, Wx, Wy
            batch_heatmaps = torch.matmul(y, x)
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]

            for pred_instances, pred_x, pred_y in zip(preds,
                                                      to_numpy(batch_pred_x),
                                                      to_numpy(batch_pred_y)):

                pred_instances.keypoint_x_labels = pred_x[None]
                pred_instances.keypoint_y_labels = pred_y[None]

            return preds, pred_fields
        else:
            return preds

    def loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        train_cfg: OptConfigType = {},
    ) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_x, pred_y = self.forward(feats)

        gt_x = torch.cat([
            d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples
        ],
                         dim=0)
        gt_y = torch.cat([
            d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples
        ],
                         dim=0)
        keypoint_weights = torch.cat(
            [
                d.gt_instance_labels.keypoint_weights
                for d in batch_data_samples
            ],
            dim=0,
        )

        pred_simcc = (pred_x, pred_y)
        gt_simcc = (gt_x, gt_y)

        # calculate losses
        losses = dict()
        loss = self.loss_module(pred_simcc, gt_simcc, keypoint_weights)

        losses.update(loss_kpt=loss)

        # calculate accuracy
        _, avg_acc, _ = simcc_pck_accuracy(
            output=to_numpy(pred_simcc),
            target=to_numpy(gt_simcc),
            simcc_split_ratio=self.simcc_split_ratio,
            mask=to_numpy(keypoint_weights) > 0,
        )

        acc_pose = torch.tensor(avg_acc, device=gt_x.device)
        losses.update(acc_pose=acc_pose)

        return losses

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(
                type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1),
            dict(type='Normal', layer=['Linear'], std=0.01, bias=0),
        ]
        return init_cfg
