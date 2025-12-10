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

import sys
import os

# Add the Depth Anything v3 directory to Python path
depth_anything_path = os.path.join(os.path.dirname(__file__), 'depth_anything_v3', 'Depth_Anything_3', 'src')
sys.path.insert(0, depth_anything_path)


from memfof.update import GMAUpdateBlock
from memfof.corr import CorrBlock
from memfof.utils.utils import coords_grid, InputPadder
from memfof.extractor import ResNetFPN16x
from memfof.layer import conv3x3
from memfof.gma import Attention
from memfof.depth_anything_v3.Depth_Anything_3.src.depth_anything_3.api import DepthAnything3
#import depth anything v3


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
        backbone: Literal["resnet18", "resnet34", "resnet50"] = "resnet34",
        depth_model: Literal["depth-anything/da3mono-large", "depth-anything/da3metric-large", "depth-anything/da3-small"] = "depth-anything/da3metric-large",
        backbone_weights: WeightsEnum = ResNet34_Weights.IMAGENET1K_V1,
        dim: int = 512,
        corr_levels: int = 4,
        corr_radius: int = 4,
        num_blocks: int = 2,
        use_var: bool = True,
        var_min: float = 0.0,
        var_max: float = 10.0,
    ):
        super().__init__()
        self.dim = dim
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.use_var = use_var
        self.var_min = var_min
        self.var_max = var_max

        # Add args for Depth Anything v3
        
        self.dav3 = DepthAnything3.from_pretrained(depth_model)
        self.cnet = ResNetFPN16x(9, dim * 2, backbone, backbone_weights)
        self.bnet = ResNetFPN16x(6, dim * 2, backbone, backbone_weights)

        self.init_conv = conv3x3(2 * dim, 2 * dim)

        self.upsample_weight = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim * 2, 2 * 16 * 16 * 9, 1, padding=0),
        )

        self.flow_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(dim, 2 * dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * dim, 2 * 6, 3, padding=1),
        )
        
        self.merge_head = nn.Sequential(
            nn.Conv2d(dim, dim//2 * 3, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//2 * 3, dim*2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, dim * 2, 3, stride=2, padding=1)
        )

        self.fnet = ResNetFPN16x(3, dim * 2, backbone, backbone_weights)

        corr_channel = corr_levels * (corr_radius * 2 + 1) ** 2
        self.update_block = GMAUpdateBlock(num_blocks, corr_channel, hdim=dim, cdim=dim)

        self.att = Attention(dim=dim, heads=1, dim_head=dim)

    def create_bases(self, disp):
        B, C, H, W = disp.shape
        assert C == 1
        cx = 0.5
        cy = 0.5

        ys = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H)
        xs = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W)
        u, v = torch.meshgrid(xs, ys, indexing='xy')
        u = u - cx
        v = v - cy
        u = u.unsqueeze(0).unsqueeze(0)
        v = v.unsqueeze(0).unsqueeze(0)
        u = u.repeat(B, 1, 1, 1).cuda()
        v = v.repeat(B, 1, 1, 1).cuda()

        aspect_ratio = W / H

        Tx = torch.cat([-torch.ones_like(disp), torch.zeros_like(disp)], dim=1)
        Ty = torch.cat([torch.zeros_like(disp), -torch.ones_like(disp)], dim=1)
        Tz = torch.cat([u, v], dim=1)

        Tx = Tx / torch.linalg.vector_norm(Tx, dim=(1,2,3), keepdim=True)
        Ty = Ty / torch.linalg.vector_norm(Ty, dim=(1,2,3), keepdim=True)
        Tz = Tz / torch.linalg.vector_norm(Tz, dim=(1,2,3), keepdim=True)
        
        Tx = 2 * disp * Tx
        Ty = 2 * disp * Ty
        Tz = 2 * disp * Tz

        R1x = torch.cat([torch.zeros_like(disp), torch.ones_like(disp)], dim=1)
        R2x = torch.cat([u * v, v * v], dim=1)
        R1y = torch.cat([-torch.ones_like(disp), torch.zeros_like(disp)], dim=1)
        R2y = torch.cat([-u * u, -u * v], dim=1)
        Rz =  torch.cat([-v / aspect_ratio, u * aspect_ratio], dim=1)

        R1x = R1x / torch.linalg.vector_norm(R1x, dim=(1,2,3), keepdim=True)
        R2x = R2x / torch.linalg.vector_norm(R2x, dim=(1,2,3), keepdim=True)
        R1y = R1y / torch.linalg.vector_norm(R1y, dim=(1,2,3), keepdim=True)
        R2y = R2y / torch.linalg.vector_norm(R2y, dim=(1,2,3), keepdim=True)
        Rz =  Rz  / torch.linalg.vector_norm(Rz,  dim=(1,2,3), keepdim=True)
        
        M = torch.cat([Tx, Ty, Tz, R1x, R2x, R1y, R2y, Rz], dim=1) # Bx(8x2)xHxW
        return M
    
    def find_depth(self, image):
        _, _, H, W = image.shape
        
        H = ((H+13)//14)*14
        W = ((W+13)//14)*14
        
        image = F.interpolate(image, size=(H, W), mode="bilinear", align_corners=False)

        predictions = self.dav3.forward(image.unsqueeze(1), export_feat_layers=[])

        # Get device from first tensor in predictions
        
        # Depth map - check different possible keys
        if "depth" in predictions:
            depth_map = predictions["depth"]  # Should be [B, H, W] or [B, N, H, W]
        else:
            raise KeyError("Depth map not found in predictions")
        
        # Features - check different possible keys
        f_map = None
        for key in ["features", "aux", "conf", "feature"]:
            if key in predictions:
                f_map = predictions[key]
                break
        
        # Ensure depth_map has proper shape [B, 1, H, W]
        if len(depth_map.shape) == 3:  # [B, H, W]
            depth_map = depth_map.unsqueeze(1)  # [B, 1, H, W]
        elif len(depth_map.shape) == 4 and depth_map.shape[1] > 1:  # [B, N, H, W], N>1
            # Take first view if multi-view
            depth_map = depth_map[:, 0:1, :, :]  # [B, 1, H, W]
        
        # Ensure f_map has proper shape if it exists
        if len(f_map.shape) == 3:  # [B, H, W]
            f_map = f_map.unsqueeze(1)  # [B, 1, H, W]
        elif len(f_map.shape) == 4 and f_map.shape[1] > 1:  # [B, N, H, W], N>1
            # Take first view if multi-view
            f_map = f_map[:, 0:1, :, :]  # [B, 1, H, W]
        
        return depth_map, f_map
        
    
    def forward(
        self,
        images: torch.Tensor,
        iters: int = 8,
        flow_gts: torch.Tensor | None = None,
        fmap_cache: list[torch.Tensor | None] = [None, None, None],
        dmap_cache: list[torch.Tensor | None] = [None, None, None],
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        """Forward pass of the MEMFOF model.

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

        images = 2 * (images / 255.0) - 1.0
        images = images.contiguous()


        self.dav3 = self.dav3.to(device=images.device)
        dmap1, fmap1 = (
                self.find_depth(images[:, 0])
                if dmap_cache[0] is None
                else dmap_cache[0].clone().to(bnet)
            )
        dmap2, fmap2 = (
                self.find_depth(images[:, 1])
                if dmap_cache[1] is None
                else dmap_cache[1].clone().to(bnet)
            )
        dmap3, fmap3 = (
                self.find_depth(images[:, 2])
                if dmap_cache[2] is None
                else dmap_cache[2].clone().to(bnet)
            )
            
        fmap1 = (F.interpolate(fmap1, (H, W), mode="bilinear", align_corners=False) if fmap1 is not None else dmap1)
        fmap2 = (F.interpolate(fmap2, (H, W), mode="bilinear", align_corners=False) if fmap2 is not None else dmap2)
        fmap3 = (F.interpolate(fmap3, (H, W), mode="bilinear", align_corners=False) if fmap3 is not None else dmap3)
        
        bases1 = self.create_bases(F.interpolate(dmap1, (H, W), mode="bilinear", align_corners=False))
        bases3 = self.create_bases(F.interpolate(dmap3, (H, W), mode="bilinear", align_corners=False))

        mono1 = self.merge_head(fmap1)
        mono2 = self.merge_head(fmap2)
        mono3 = self.merge_head(fmap3)
            
        
        flow_predictions = []
        info_predictions = []

        # padding
        padder = InputPadder(images.shape)
        images = padder.pad(images)
        bases1 = padder.pad(bases1)
        bases3 = padder.pad(bases3)
        
        B, _, _, H, W = images.shape
        dilation = torch.ones(B, 1, H // 16, W // 16, device=images.device)

        # run the context network
        cnet = self.cnet(torch.cat([images[:, 0], images[:, 1], images[:, 2]], dim=1))
        cnet = self.init_conv(cnet)
        net, context = torch.split(cnet, [self.dim, self.dim], dim=1)
        attention = self.att(context)
        
        # Run base_net
        bnet = self.bnet(torch.cat([bases1, bases3], dim=1))
        bnet = self.init_conv(bnet)
        netbases, ctxbases = torch.split(bnet, [self.dim, self.dim], dim=1)
        
        net = torch.cat((net, netbases), 1)
        context = torch.cat((context, ctxbases), 1)

        # init flow
        flow_update = self.flow_head(net)
        weight_update = 0.25 * self.upsample_weight(net)

        flow_16x_21 = flow_update[:, 0:2]
        info_16x_21 = flow_update[:, 2:6]

        flow_16x_23 = flow_update[:, 6:8]
        info_16x_23 = flow_update[:, 8:12]

        if self.training or iters == 0:
            flow_up_21, info_up_21 = self._upsample_data(
                flow_16x_21, info_16x_21, weight_update[:, : 16 * 16 * 9]
            )
            flow_up_23, info_up_23 = self._upsample_data(
                flow_16x_23, info_16x_23, weight_update[:, 16 * 16 * 9 :]
            )
            flow_predictions.append(torch.stack([flow_up_21, flow_up_23], dim=1))
            info_predictions.append(torch.stack([info_up_21, info_up_23], dim=1))

        if iters > 0:
            # run the feature network
            fmap1_16x = (
                self.fnet(images[:, 0])
                if fmap_cache[0] is None
                else fmap_cache[0].clone().to(cnet)
            )
            fmap2_16x = (
                self.fnet(images[:, 1])
                if fmap_cache[1] is None
                else fmap_cache[1].clone().to(cnet)
            )
            fmap3_16x = (
                self.fnet(images[:, 2])
                if fmap_cache[2] is None
                else fmap_cache[2].clone().to(cnet)
            )
            # Run depth extraction network, use dmap_cache
            fmap1_16x = torch.cat((fmap1_16x, mono1), 1)
            fmap2_16x = torch.cat((fmap2_16x, mono2), 1)
            fmap3_16x = torch.cat((fmap3_16x, mono3), 1)
            
            corr_fn_21 = CorrBlock(
                fmap2_16x, fmap1_16x, self.corr_levels, self.corr_radius
            )
            corr_fn_23 = CorrBlock(
                fmap2_16x, fmap3_16x, self.corr_levels, self.corr_radius
            )

        for itr in range(iters):
            B, _, H, W = flow_16x_21.shape
            flow_16x_21 = flow_16x_21.detach()
            flow_16x_23 = flow_16x_23.detach()

            coords21 = (
                coords_grid(B, H, W, device=images.device) + flow_16x_21
            ).detach()
            coords23 = (
                coords_grid(B, H, W, device=images.device) + flow_16x_23
            ).detach()

            corr_21 = corr_fn_21(coords21, dilation=dilation)
            corr_23 = corr_fn_23(coords23, dilation=dilation)

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
                flow_predictions.append(torch.stack([flow_up_21, flow_up_23], dim=1))
                info_predictions.append(torch.stack([info_up_21, info_up_23], dim=1))

        for i in range(len(info_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            info_predictions[i] = padder.unpad(info_predictions[i])

        new_fmap_cache = [None, None, None]
        if iters > 0:
            new_fmap_cache[0] = fmap1_16x.clone().cpu()
            new_fmap_cache[1] = fmap2_16x.clone().cpu()
            new_fmap_cache[2] = fmap3_16x.clone().cpu()
            
        new_dmap_cache = [(None,None), (None,None), (None,None)]
        if iters > 0:
            new_dmap_cache[0] = dmap1.clone().cpu(), fmap1.clone().cpu()
            new_dmap_cache[1] = dmap2.clone().cpu(), fmap2.clone().cpu()
            new_dmap_cache[2] = dmap3.clone().cpu(), fmap2.clone().cpu()

        if not self.training:
            return {
                "flow": flow_predictions,
                "info": info_predictions,
                "nf": None,
                "fmap_cache": new_fmap_cache,
                "dmap_cache": new_dmap_cache,
            }
        else:
            # exlude invalid pixels and extremely large diplacements
            nf_predictions = []
            for i in range(len(info_predictions)):
                if not self.use_var:
                    var_max = var_min = 0
                else:
                    var_max = self.var_max
                    var_min = self.var_min

                nf_losses = []
                for k in range(2):
                    raw_b = info_predictions[i][:, k, 2:]
                    log_b = torch.zeros_like(raw_b)
                    weight = info_predictions[i][:, k, :2]
                    # Large b Component
                    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
                    # Small b Component
                    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
                    # term2: [N, 2, m, H, W]
                    term2 = (
                        (flow_gts[:, k] - flow_predictions[i][:, k]).abs().unsqueeze(2)
                    ) * (torch.exp(-log_b).unsqueeze(1))
                    # term1: [N, m, H, W]
                    term1 = weight - math.log(2) - log_b
                    nf_loss = torch.logsumexp(
                        weight, dim=1, keepdim=True
                    ) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
                    nf_losses.append(nf_loss)

                nf_predictions.append(torch.stack(nf_losses, dim=1))

            return {
                "flow": flow_predictions,
                "info": info_predictions,
                "nf": nf_predictions,
                "fmap_cache": new_fmap_cache,
                "dmap_cache": new_dmap_cache,
            }

    def _get_depth(self, img):
        """
        Args: image tensor
        
        output: Depth map,
                Feature map
        """
        
        img_np = img.permute(0, 2, 3, 1).cpu().numpy()
        
        all_depths = []
        for i in range(img_np.shape[0]):
            pred = self.dav3.forward()
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
        
