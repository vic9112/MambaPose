import numpy as np
import torch
import torch.nn as nn
import copy
import math
import warnings
# from mmcv.cnn import build_upsample_layer, Linear, bias_init_with_prob, constant_init, normal_init
from mmengine.model import normal_init
import torch.nn.functional as F

# from mmpose.core.evaluation import (keypoint_pck_accuracy,keypoints_from_regression)
# from mmpose.models.builder import build_loss, build_transformer
from mmcv.cnn import Conv2d, build_activation_layer
from mmcv.cnn.bricks.transformer import Linear, FFN, build_positional_encoding
from mmcv.cnn import ConvModule
import torch.distributions as distributions
from .rle_regression_head import nets, nett, RealNVP
from easydict import EasyDict
from mmpose.models.losses.regression_loss import L1Loss
from mmpose.models.losses.rle_loss import RLELoss_poseur, RLEOHKMLoss

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from mmpose.evaluation.functional import keypoint_pck_accuracy
from mmpose.models.utils.tta import flip_coordinates
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, OptConfigType, OptSampleList,
                                 Predictions)
from ..base_head import BaseHead


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


def fliplr_rle_regression(regression,
                          regression_score,
                          flip_pairs,
                          center_mode='static',
                          center_x=0.5,
                          center_index=0):
    """Flip human joints horizontally.

    Note:
        batch_size: N
        num_keypoint: K
    Args:
        regression (np.ndarray([..., K, C])): Coordinates of keypoints, where K
            is the joint number and C is the dimension. Example shapes are:
            - [N, K, C]: a batch of keypoints where N is the batch size.
            - [N, T, K, C]: a batch of pose sequences, where T is the frame
                number.
        flip_pairs (list[tuple()]): Pairs of keypoints which are mirrored
            (for example, left ear -- right ear).
        center_mode (str): The mode to set the center location on the x-axis
            to flip around. Options are:
            - static: use a static x value (see center_x also)
            - root: use a root joint (see center_index also)
        center_x (float): Set the x-axis location of the flip center. Only used
            when center_mode=static.
        center_index (int): Set the index of the root joint, whose x location
            will be used as the flip center. Only used when center_mode=root.

    Returns:
        tuple: Flipped human joints.

        - regression_flipped (np.ndarray([..., K, C])): Flipped joints.
    """
    assert regression.ndim >= 2, f'Invalid pose shape {regression.shape}'

    allowed_center_mode = {'static', 'root'}
    assert center_mode in allowed_center_mode, 'Get invalid center_mode ' \
                                               f'{center_mode}, allowed choices are {allowed_center_mode}'

    if center_mode == 'static':
        x_c = center_x
    elif center_mode == 'root':
        assert regression.shape[-2] > center_index
        x_c = regression[..., center_index:center_index + 1, 0]

    regression_flipped = regression.copy()
    regression_score_flipped = regression_score.copy()

    # Swap left-right parts
    for left, right in flip_pairs:
        regression_flipped[..., left, :] = regression[..., right, :]
        regression_flipped[..., right, :] = regression[..., left, :]
        regression_score_flipped[..., left, :] = regression_score[..., right, :]
        regression_score_flipped[..., right, :] = regression_score[..., left, :]

    # Flip horizontally
    regression_flipped[..., 0] = x_c * 2 - regression_flipped[..., 0]
    return regression_flipped, regression_score_flipped


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
            x_norm = torch.norm(x, dim=-1, keepdim=True)
            y = y / x_norm

        if self.bias:
            y = y + self.linear.bias
        return y


@MODELS.register_module()
class PoseurHead(BaseHead):
    """
    rle loss for Poseur
    """

    def __init__(self,
                 in_channels,
                 num_queries=17,
                 num_reg_fcs=2,
                 decoder: OptConfigType = None,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 transformer=None,
                 with_box_refine=False,
                 as_two_stage=False,
                 heatmap_size=[64, 48],
                 num_joints=17,
                 loss_coord_enc=None,
                 loss_coord_dec=None,
                 loss_hp_keypoint=None,
                 use_heatmap_loss=True,
                 train_cfg=None,
                 test_cfg=None,
                 use_udp=False,
                 ):
        super().__init__()
        self.use_udp = use_udp
        self.num_queries = num_queries
        self.num_reg_fcs = num_reg_fcs
        self.in_channels = in_channels
        self.act_cfg = transformer.get('act_cfg', dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        self.transformer = MODELS.build(transformer)
        self.embed_dims = self.transformer.embed_dims
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
                                                 f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
                                                 f' and {num_feats}.'

        self.num_joints = num_joints
        self.heatmap_size = heatmap_size
        self.loss_coord_enc = MODELS.build(loss_coord_enc)
        self.loss_coord_dec = MODELS.build(loss_coord_dec)

        self.use_dec_rle_loss = isinstance(self.loss_coord_dec, RLELoss_poseur) or isinstance(self.loss_coord_dec,
                                                                                              RLEOHKMLoss)

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg

        masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
        enc_prior = distributions.MultivariateNormal(torch.zeros(2) + 0.5, torch.eye(2))
        self.enc_flow = RealNVP(nets, nett, masks, enc_prior)

        if self.use_dec_rle_loss:
            dec_prior = distributions.MultivariateNormal(torch.zeros(2) + 0.5, torch.eye(2))
            self.dec_flow = RealNVP(nets, nett, masks, dec_prior)

        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""

        fc_coord_branch = []
        for _ in range(self.num_reg_fcs):
            fc_coord_branch.append(Linear(self.embed_dims, self.embed_dims))
            fc_coord_branch.append(nn.ReLU())
        # fc_coord_branch.append(Linear_with_norm(self.embed_dims, 2))
        fc_coord_branch.append(Linear(self.embed_dims, 2))
        fc_coord_branch = nn.Sequential(*fc_coord_branch)

        if self.use_dec_rle_loss:
            fc_sigma_branch = []
            for _ in range(self.num_reg_fcs):
                fc_sigma_branch.append(Linear(self.embed_dims, self.embed_dims))
                # fc_sigma_branch.append(nn.ReLU())
            fc_sigma_branch.append(Linear_with_norm(self.embed_dims, 2, norm=False))
            #         fc_sigma_branch.append(Linear(self.embed_dims, 2))
            fc_sigma_branch = nn.Sequential(*fc_sigma_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = self.transformer.decoder.num_layers
        # num_pred = 6

        if self.with_box_refine:
            self.fc_coord_branches = _get_clones(fc_coord_branch, num_pred)
            self.fc_coord_output_branches = _get_clones(fc_coord_branch, num_pred)
            if self.use_dec_rle_loss:
                self.fc_sigma_branches = _get_clones(fc_sigma_branch, num_pred)
            # if self.as_two_stage:
            #     self.cls_out_channels = 2
            #     fc_cls = Linear(self.embed_dims, self.cls_out_channels)
            #     self.cls_branches = _get_clones(fc_cls, num_pred)
        else:
            self.fc_coord_branches = nn.ModuleList(
                [fc_coord_branch for _ in range(num_pred)])
            if isinstance(self.loss_coord_dec, RLELoss) or isinstance(self.loss_coord_dec, RLEOHKMLoss):
                self.fc_sigma_branches = nn.ModuleList([fc_sigma_branch for _ in range(1)])

        if self.as_two_stage:
            self.query_embedding = None
        else:
            self.query_embedding = nn.Embedding(self.num_queries,
                                                self.embed_dims * 2)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()

        # for m in [self.fc_coord_branches, self.fc_sigma_branches]:
        for m in [self.fc_coord_branches]:
            for mm in m:
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)

        for m in [self.fc_coord_output_branches]:
            for mm in m:
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)

    def forward(self, mlvl_feats):
        # print("mlvl_feats[0].shape", mlvl_feats[0].shape)
        # print("mlvl_feats[1].shape", mlvl_feats[1].shape)
        # print("mlvl_feats[2].shape", mlvl_feats[2].shape)
        # print("mlvl_feats[3].shape", mlvl_feats[3].shape)
        batch_size = mlvl_feats[0].size(0)
        # input_img_h, input_img_w, _ = img_metas[0]['img_shape']
        img_w, img_h = self.train_cfg['image_size']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, img_h, img_w))
        for img_id in range(batch_size):
            # img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight

        memory, spatial_shapes, level_start_index, hs, init_reference, inter_references, \
            enc_outputs = self.transformer(
            mlvl_feats,
            mlvl_masks,
            query_embeds,
            mlvl_positional_encodings,
            reg_branches=self.fc_coord_branches if self.with_box_refine else None,  # noqa:E501
            # cls_branches=self.cls_branches if self.as_two_stage else None  # noqa:E501
            cls_branches=None  # noqa:E501
        )
        # hs = hs.permute(0, 2, 1, 3)
        outputs_coords = []
        outputs_sigmas = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            # hs: lvl, bs, 17, 256
            tmp = self.fc_coord_branches[lvl](hs[lvl])
            # print("tmp.shape", tmp.shape)
            # print("reference.shape", reference.shape)
            tmp[..., :2] += reference

            if self.use_dec_rle_loss:
                outputs_sigma = self.fc_sigma_branches[lvl](hs[lvl])
                outputs_sigmas.append(outputs_sigma)

            outputs_coord = tmp.sigmoid()
            delta_coord_output = self.fc_coord_output_branches[lvl](hs[lvl])
            outputs_coord = outputs_coord + delta_coord_output
            outputs_coords.append(outputs_coord)

        # [num_decoder, bs, 17, 2]
        if self.with_box_refine:
            outputs_coords = torch.stack(outputs_coords)
            if isinstance(self.loss_coord_dec, RLELoss_poseur) or isinstance(self.loss_coord_dec, RLEOHKMLoss):
                outputs_sigmas = torch.stack(outputs_sigmas).sigmoid()
                # (B, N, 2)
                scores = 1 - outputs_sigmas[-1]
                # (B, N, 1)
                scores = torch.mean(scores, dim=-1, keepdim=True)

        dec_outputs = EasyDict(pred_jts=outputs_coords)

        if self.use_dec_rle_loss:
            dec_outputs.sigma = outputs_sigmas
            dec_outputs.maxvals = scores.float()

        return enc_outputs, dec_outputs

    def get_loss(self, enc_output, dec_output, coord_target, coord_target_weight, hp_target, hp_target_weight):
        losses = dict()
        if self.as_two_stage and enc_output is not None:
            enc_rle_loss = self.get_enc_rle_loss(enc_output, coord_target, coord_target_weight)
            losses.update(enc_rle_loss)

        dec_rle_loss = self.get_dec_rle_loss(dec_output, coord_target, coord_target_weight)
        losses.update(dec_rle_loss)

        return losses

    def loss(self,
             inputs: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        enc_outputs, dec_outputs = self.forward(inputs)
        keypoint_labels = torch.cat(
            [d.gt_instance_labels.keypoint_labels for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])
        keypoint_weights_2 = keypoint_weights.unsqueeze(-1).repeat(1, 1, 2)
        # print("keypoint_labels",keypoint_labels.shape)
        # print("keypoint_weights", keypoint_weights.shape)
        losses = dict()
        enc_rle_loss = self.get_enc_rle_loss(enc_outputs, keypoint_labels, keypoint_weights_2)
        losses.update(enc_rle_loss)
        dec_rle_loss = self.get_dec_rle_loss(dec_outputs, keypoint_labels, keypoint_weights_2)
        losses.update(dec_rle_loss)
        # calculate accuracy
        # print(dec_outputs.pred_jts[0].shape)
        # print(keypoint_labels.shape)

        pre_points = dec_outputs.pred_jts[-1]
        pre_points = pre_points[:, 0:17, :]
        # print(pre_points.shape)
        _, avg_acc, _ = keypoint_pck_accuracy(
            pred=to_numpy(pre_points),
            gt=to_numpy(keypoint_labels),
            mask=to_numpy(keypoint_weights) > 0,
            thr=0.05,
            norm_factor=np.ones((pre_points.size(0), 2), dtype=np.float32))

        acc_pose = torch.tensor(avg_acc, device=keypoint_labels.device)
        losses.update(acc_pose=acc_pose)

        return losses

    def get_enc_rle_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.
        Note:
            batch_size: N
            num_keypoints: K
        Args:
            output (torch.Tensor[N, K, 2]): Output keypoints.
            target (torch.Tensor[N, K, 2]): Target keypoints.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """

        losses = dict()
        assert not isinstance(self.loss_coord_enc, nn.Sequential)
        assert target.dim() == 3 and target_weight.dim() == 3

        BATCH_SIZE = output.sigma.size(0)
        gt_uv = target.reshape(output.pred_jts.shape)
        bar_mu = (output.pred_jts - gt_uv) / output.sigma
        # (B, K, 1)
        log_phi = self.enc_flow.log_prob(bar_mu.reshape(-1, 2)).reshape(BATCH_SIZE, self.num_joints, 1)
        output.nf_loss = torch.log(output.sigma) - log_phi
        losses['enc_rle_loss'] = self.loss_coord_enc(output, target, target_weight)

        return losses

    def get_dec_rle_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2]): Output keypoints.
            target (torch.Tensor[N, K, 2]): Target keypoints.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """

        losses = dict()
        assert not isinstance(self.loss_coord_dec, nn.Sequential)
        assert target.dim() == 3 and target_weight.dim() == 3
        target = target.repeat(1, self.transformer.num_noise_sample + 1, 1)
        target_weight = target_weight.repeat(1, self.transformer.num_noise_sample + 1, 1)

        if self.with_box_refine:
            if self.use_dec_rle_loss:
                for i, (pred_jts, sigma) in enumerate(zip(output.pred_jts, output.sigma)):
                    output_i = EasyDict(
                        pred_jts=pred_jts,
                        sigma=sigma
                    )
                    BATCH_SIZE = output_i.sigma.size(0)
                    gt_uv = target.reshape(output_i.pred_jts.shape)
                    bar_mu = (output_i.pred_jts - gt_uv) / output_i.sigma
                    # (B, K, 1)
                    log_phi = self.dec_flow.log_prob(bar_mu.reshape(-1, 2)).reshape(BATCH_SIZE,
                                                                                    self.num_joints * (
                                                                                                self.transformer.num_noise_sample + 1),
                                                                                    1)
                    output_i.nf_loss = torch.log(output_i.sigma) - log_phi
                    losses['dec_rle_loss_{}'.format(i)] = self.loss_coord_dec(output_i, target, target_weight)
            else:
                for i, pred_jts in enumerate(output.pred_jts):
                    losses['dec_rle_loss_{}'.format(i)] = self.loss_coord_dec(pred_jts, target, target_weight)
        else:
            if self.use_dec_rle_loss:
                BATCH_SIZE = output.sigma.size(0)
                gt_uv = target.reshape(output.pred_jts.shape)
                bar_mu = (output.pred_jts - gt_uv) / output.sigma
                # (B, K, 1)
                log_phi = self.dec_flow.log_prob(bar_mu.reshape(-1, 2)).reshape(BATCH_SIZE, self.num_joints, 1)
                output.nf_loss = torch.log(output.sigma) - log_phi
                losses['dec_rle_loss'] = self.loss_coord_dec(output, target, target_weight) * 0
            else:
                losses['dec_rle_loss'] = self.loss_coord_dec(output.pred_jts, target + 0.5, target_weight) * 0

        return losses

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """Predict results from outputs."""

        if 1:
            # print(1)
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            input_size = batch_data_samples[0].metainfo['input_size']

            _feats, _feats_flip = feats

            output_regression, output_regression_score = self.inference_model(_feats, flip_pairs=None)
            output_regression_score, output_regression_sigma = self.seperate_sigma_from_score(output_regression_score)
            flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
            output_regression_flipped, output_regression_score_flipped = self.inference_model(_feats_flip,flip_pairs )
            output_regression_score_flipped, output_regression_sigma_flipped = self.seperate_sigma_from_score(output_regression_score_flipped)

            output_regression, output_regression_flipped = \
                torch.from_numpy(output_regression), torch.from_numpy(output_regression_flipped)

            output_regression_sigma, output_regression_sigma_flipped = \
                torch.from_numpy(output_regression_sigma), torch.from_numpy(output_regression_sigma_flipped)

            output_regression_p, output_regression_p_flipped = \
                self.get_p(output_regression_sigma), self.get_p(output_regression_sigma_flipped)

            p_to_coord_index = 5
            output_regression = (output_regression * output_regression_p ** p_to_coord_index + output_regression_flipped * output_regression_p_flipped ** p_to_coord_index) \
                                / (output_regression_p ** p_to_coord_index + output_regression_p_flipped ** p_to_coord_index + 1e-10)

            output_regression_score = (output_regression_p + output_regression_p_flipped) * 0.5

        else:
            output_regression, output_regression_score = self.inference_model(feats, flip_pairs=None)
            output_regression_score, output_regression_sigma = self.seperate_sigma_from_score(output_regression_score)

        output_regression = output_regression.unsqueeze_(dim=1)  # (B, N, K, D)
        # print(output_regression.shape)
        preds = self.decode(output_regression)

        # enc_outputs, dec_outputs = self.forward(feats)
        # batch_coords = dec_outputs.pred_jts[-1]  # (B, K, D)
        # batch_coords[..., 2:] = batch_coords[..., 2:].sigmoid()
        #
        # batch_coords.unsqueeze_(dim=1)  # (B, N, K, D)
        # print(batch_coords.shape)
        # preds = self.decode(batch_coords)

        return preds

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_regression (np.ndarray): Output regression.

        Args:
            x (torch.Tensor[N, K, 2]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """
        output_enc, output_dec = self.forward(x)
        # coord_output = output["coord"]
        # output_regression, output_regression_score = output_enc.pred_jts.detach().cpu().numpy(), output_enc.maxvals.detach().cpu().numpy()
        output_regression, output_regression_score = output_dec.pred_jts.detach().cpu().numpy(), output_dec.maxvals.detach().cpu().numpy()
        # hp_output = output["hp"]
        output_sigma = output_dec.sigma.detach().cpu().numpy()
        output_sigma = output_sigma[-1]
        output_regression_score = np.concatenate([output_regression_score, output_sigma], axis=2)

        if output_regression.ndim == 4:
            output_regression = output_regression[-1]

        if flip_pairs is not None:
            output_regression, output_regression_score = fliplr_rle_regression(
                output_regression, output_regression_score, flip_pairs)

        return output_regression, output_regression_score

    def decode_keypoints(self, img_metas, output_regression, output_regression_score, img_size):
        """Decode keypoints from output regression.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output_regression (np.ndarray[N, K, 2]): model
                predicted regression vector.
            img_size (tuple(img_width, img_height)): model input image size.
        """
        batch_size = len(img_metas)

        if 'bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i]['center']
            s[i, :] = img_metas[i]['scale']
            image_paths.append(img_metas[i]['image_file'])

            if 'bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)

            if bbox_ids is not None:
                bbox_ids.append(img_metas[i]['bbox_id'])

        preds, maxvals = keypoints_from_regression(output_regression, c, s,
                                                   img_size)

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        # all_preds[:, :, 2:3] = maxvals
        all_preds[:, :, 2:3] = output_regression_score
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score

        result = {}

        result['preds'] = all_preds
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids

        return result

    def seperate_sigma_from_score(self, score):
        if score.shape[2] == 3:
            sigma = score[:,:,[1,2]]
            score = score[:,:,[0]]
            return score, sigma
        elif score.shape[2] == 1:
            return score, None
        else:
            raise

    def get_p(self, output_regression_sigma, p_x=0.2):
        output_regression_p = (1 - np.exp(-(p_x / output_regression_sigma)))
        output_regression_p = output_regression_p[:, :, 0] * output_regression_p[:, :, 1]
        output_regression_p = output_regression_p[:, :, None]
        return output_regression_p * 0.7