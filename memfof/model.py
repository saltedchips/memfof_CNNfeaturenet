from typing import Literal
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torchvision.models import (
    ResNet34_Weights,
    WeightsEnum,
)

from memfof.update import GMAUpdateBlock
from memfof.corr import CorrBlock
from memfof.utils.utils import coords_grid, InputPadder
from memfof.extractor import ResNetFPN16x
from memfof.layer import conv3x3
from memfof.gma import Attention
from memfof.backbone2 import CNNEncoder


AVAILABLE_MODELS = [
    "MEMFOF-Tartan",
    "MEMFOF-Tartan-T",
    "MEMFOF-Tartan-T-TSKH",
    "MEMFOF-Tartan-T-TSKH-kitti",
    "MEMFOF-Tartan-T-TSKH-sintel",
    "MEMFOF-Tartan-T-TSKH-spring",
]


class MEMFOF(
    nn.Module,
    PyTorchModelHubMixin,
    # optionally, you can add metadata which gets pushed to the model card
    pipeline_tag="optical-flow-estimation",
    license="bsd-3-clause",
):
    def __init__(
        self,
        dim: int = 512,
        backbone: Literal['CNNEncoder', 'resnet', 'mobilenet'] = 'resnet',
        corr_levels: int = 4,
        corr_radius: int = 4,
        num_blocks: int = 2,
        use_var: bool = True,
        var_min: float = 0.0,
        var_max: float = 10.0,
        backbone_weights: WeightsEnum = ResNet34_Weights.IMAGENET1K_V1,
        ):
        
        super().__init__()
        self.dim = dim
        if backbone == 'CNNEncoder':
            self.cnet = CNNEncoder(dim)
        elif backbone[6:] == 'resnet':
            self.cnet = ResNetFPN16x(9, dim * 2, backbone, backbone_weights)
        else:
            self.cnet = ResNetFPN16x(9, dim * 2, backbone, backbone_weights)
        
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.num_blocks = num_blocks
        self.use_var = use_var
        self.var_min = var_min
        self.var_max = var_max
        
        self.init_conv = conv3x3(dim*2, dim*2)
        
        self.upsample_weight = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim * 2, 2 * 16 * 16 * 9, 1, padding=0),
        )

        self.merge_s16 = nn.Sequential(
            nn.Conv2d(128*2, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128)
        )
            

        self.flow_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(dim, 2 * dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * dim, 2 * 6, 3, padding=1),
        )

        self.fnet = CNNEncoder(128)
        
        corr_channel = corr_levels * (corr_radius * 2 + 1) **2
        self.update_block = GMAUpdateBlock(num_blocks, corr_channel, hdim=dim, cdim=dim)

        self.att = Attention(dim=dim, heads=1, dim_head=dim)
        
    def init_bhw(self, batch_size, height, width):
        self.backbone.init_pos_12(batch_size, height//8, width//8)
        
    def forward(
            self,
            images: torch.Tensor,
            iters: int = 8,
            flow_gts: torch.Tensor | None = None,
            fmap_cache: list[torch.Tensor | None] = [None, None, None],
        ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        
        """Forward pass of the RealFlow model.
        Parameters
        ----------
        images : torch.Tensor
            Tensor of shape [B, 3, 3, H, W].
            Images should be in range [0, 255].
        iters : int, optional
            Number of iterations for flow refinement, by default 8
        flow_gts : torch.Tensor | None, optional
            Ground truth flow fields of shape [B, 2, 2, H, W], by default None
           First dimension of size 2 represents backward and forward flows
            Second dimension of size 2 represents x and y components
        fmap_cache : list[torch.Tensor | None], optional
            Cache for feature maps to be used in current forward pass, by default [None, None, None]

        Returns
        -------
        dict[str, torch.Tensor | list[torch.Tensor] | None]
            Dictionary containing:
            - "flow": List of flow predictions of shape [B, 2, 2, H, W] at each iteration
            - "info": List of additional information of shape [B, 2, 4, H, W] at each iteration
            - "nf": List of negative free energy losses of shape [B, 2, 2, H, W] at each iteration (only during training)
            - "fmap_cache": Feature map cache of this forward pass
        """
            
        B, _, _, H, W = images.shape
        if flow_gts is None:
            flow_gts = torch.zeros(B, 2, 2, H, W, device=images.device)
            
        images = 2 * (images / 255.0) - 1.0  # normalize to [-1, 1]
        images = images.contiguous()
        
        flow_pred = []
        info_pred = []
        
        # Padding
        padder = InputPadder(images.shape)
        images = padder.pad(images)
        B, _, _, H, W = images.shape
        diliation = torch.ones(B, 1, H // 16, W // 16, device=images.device)

        # Context network
        cnet = self.cnet(torch.cat([images[:, 0], images[:, 1], images[:, 2]], dim=1))
        cnet = self.init_conv(cnet)
        net, context = torch.split(cnet, [self.dim, self.dim], dim=1)
        attention = self.att(context)
        
        # Init Flow
        self.fnet.init_pos(B, H, W)
        self.fnet.init_pos_12(B, H, W)
        
        flow_update = self.flow_head(net)
        update_weight = .25 * self.upsample_weight(net)
        
        flow_16x_21 = flow_update[:, 0:2]
        info_16x_21 = flow_update[:, 2:6]

        flow_16x_23 = flow_update[:, 6:8]
        info_16x_23 = flow_update[:, 8:12]
        
        if self.training or iters == 0:
            flow_up_21, info_up_21 = self.upsample_flow(flow_16x_21, info_16x_21, update_weight[:, : 16 * 16 * 9])
            flow_up_23, info_up_23 = self.upsample_flow(flow_16x_23, info_16x_23, update_weight[:, 16 * 16 * 9 : ])
            flow_pred.append(torch.stack([flow_up_21, flow_up_23], dim=1))
            info_pred.append(torch.stack([info_up_21, info_up_23], dim=1))
            
        if iters > 0:
            
            # Run feature network to compute feature maps
            f1_16x, f1_32x = (self.fnet(images[:, 0])
                if fmap_cache[0] is None
                else [t.clone().to(images.device) for t in fmap_cache[0]])
            f2_16x, f2_32x = (self.fnet(images[:, 1])
                if fmap_cache[1] is None
                else [t.clone().to(images.device) for t in fmap_cache[1]])
            f3_16x, f3_32x = (self.fnet(images[:, 2])
                if fmap_cache[2] is None
                else [t.clone().to(images.device) for t in fmap_cache[2]])
            
            H = max(f1_16x.shape[2], 2)
            W = max(f1_16x.shape[3], 2)
            
            f1_32x = F.adaptive_avg_pool2d(f1_32x, (H, W)) if fmap_cache[0] is None else f1_32x
            f2_32x = F.adaptive_avg_pool2d(f2_32x, (H, W)) if fmap_cache[1] is None else f1_32x
            f3_32x = F.adaptive_avg_pool2d(f3_32x, (H, W)) if fmap_cache[2] is None else f1_32x
            

            f1_16x = self.merge_s16(torch.cat([f1_16x, f1_32x], dim=1))
            f2_16x = self.merge_s16(torch.cat([f2_16x, f2_32x], dim=1))
            f3_16x = self.merge_s16(torch.cat([f3_16x, f3_32x], dim=1))
            
            print(f1_16x.shape)
            print(f2_16x.shape)
            print(f3_16x.shape)
            
            corr_fn_21 = CorrBlock(
                f2_16x, f1_16x, self.corr_levels, self.corr_radius
            )
            corr_fn_23 = CorrBlock(
                f2_16x, f3_16x, self.corr_levels, self.corr_radius
            )
            
        for itr in range(iters):
            B, _, H, W = flow_16x_21.shape
            flow_16x_21 = flow_16x_21.detach()
            flow_16x_23 = flow_16x_23.detach()

            # Compute flow coordinates in image space
            coords21 = (coords_grid(B, H, W, device=images.device) + flow_16x_21).detach()
            coords23 = (coords_grid(B, H, W, device=images.device) + flow_16x_23).detach()

            # --- Rescale inside CorrBlock to match feature map size ---
            corr_21 = corr_fn_21(coords21, dilation=diliation)
            corr_23 = corr_fn_23(coords23, dilation=diliation)

            corr = torch.cat([corr_21, corr_23], dim=1)
            flow_16x = torch.cat([flow_16x_21, flow_16x_23], dim=1)

            net = self.update_block(net, context, corr, flow_16x, attention)

            flow_update = self.flow_head(net)
            weight_update = 0.25 * self.upsample_weight(net)

            flow_16x_21 = flow_16x_21 + flow_update[:, 0:2]
            info_16x_21 = flow_update[:, 2:6]

            flow_16x_23 = flow_16x_23 + flow_update[:, 6:8]
            info_16x_23 = flow_update[:, 8:12]

            if self.training or itr == iters - 1:
                flow_up_21, info_up_21 = self._upsample_data(
                    flow_16x_21, info_16x_21, weight_update[:, : 16 * 16 * 9]
                )
                flow_up_23, info_up_23 = self._upsample_data(
                    flow_16x_23, info_16x_23, weight_update[:, 16 * 16 * 9 :]
                )
                flow_pred.append(torch.stack([flow_up_21, flow_up_23], dim=1))
                info_pred.append(torch.stack([info_up_21, info_up_23], dim=1))


        for i in range(len(info_pred)):
            flow_pred[i] = padder.unpad(flow_pred[i])
            info_pred[i] = padder.unpad(info_pred[i])

        new_fmap_cache = [None, None, None]
        if iters > 0:
            new_fmap_cache[0] = f1_16x.clone().cpu(), f1_32x.clone().cpu()
            new_fmap_cache[1] = f2_16x.clone().cpu(), f2_32x.clone().cpu()
            new_fmap_cache[2] = f3_16x.clone().cpu(), f3_32x.clone().cpu()

        if not self.training:
            return {
                "flow": flow_pred,
                "info": info_pred,
                "nf": None,
                "fmap_cache": new_fmap_cache,
            }
        else:
            # exlude invalid pixels and extremely large diplacements
            nf_predictions = []
            for i in range(len(info_pred)):
                if not self.use_var:
                    var_max = var_min = 0
                else:
                    var_max = self.var_max
                    var_min = self.var_min

                nf_losses = []
                for k in range(2):
                    raw_b = info_pred[i][:, k, 2:]
                    log_b = torch.zeros_like(raw_b)
                    weight = info_pred[i][:, k, :2]
                    # Large b Component
                    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
                    # Small b Component
                    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
                    # term2: [N, 2, m, H, W]
                    term2 = (
                        (flow_gts[:, k] - flow_pred[i][:, k]).abs().unsqueeze(2)
                    ) * (torch.exp(-log_b).unsqueeze(1))
                    # term1: [N, m, H, W]
                    term1 = weight - math.log(2) - log_b
                    nf_loss = torch.logsumexp(
                        weight, dim=1, keepdim=True
                    ) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
                    nf_losses.append(nf_loss)

                nf_predictions.append(torch.stack(nf_losses, dim=1))

            return {
                "flow": flow_pred,
                "info": info_pred,
                "nf": nf_predictions,
                "fmap_cache": new_fmap_cache,
            }

    def _upsample_data(self, flow, info, mask):
        """Upsample [H/16, W/16, C] -> [H, W, C] using convex combination"""
        """Forward pass of the MEMFOF model.

        Parameters
        ----------
        flow : torch.Tensor
            Tensor of shape [B, 2, H / 16, W / 16].
        info : torch.Tensor
            Tensor of shape [B, 4, H / 16, W / 16].
        mask : torch.Tensor
            Tensor of shape [B, 9 * 16 * 16, H / 16, W / 16]
        Returns
        -------
        flow : torch.Tensor
            Tensor of shape [B, 2, H, W]
        info : torch.Tensor
            Tensor of shape [B, 4, H, W]
        """
        B, C, H, W = info.shape
        mask = mask.view(B, 1, 9, 16, 16, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(16 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(B, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(B, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(B, 2, 16 * H, 16 * W), up_info.reshape(
            B, C, 16 * H, 16 * W
        )
