import torch
import torch.nn.functional as F
import math
from memfof.utils.utils import bilinear_sampler


def coords_feature(fmap, b, x, y):
    H, W = fmap.shape[2:]
    mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    b = b.long()
    x = torch.clamp(x, 0, W - 1).long()
    y = torch.clamp(y, 0, H - 1).long()
    res = fmap[b, :, y, x] * mask.float().unsqueeze(1)
    return res


class CorrBlock:
    def __init__(self, fmap1, fmap2, corr_levels, corr_radius):
        self.num_levels = corr_levels
        self.radius = corr_radius
        b, c, h, w = fmap1.shape
        f1 = fmap1.view(b, c, h*w)
        f2 = fmap2.view(b, c, h*w)
        
        corr = torch.matmul(f1.transpose(1,2), f2)
        corr = corr.view(b*h*w, 1, h, w) / math.sqrt(c)

        self.corr_pyramid = [corr]
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, kernel_size=2, stride=2)
            self.corr_pyramid.append(corr)


    def __call__(self, coords, dilation=None):
        """
        Args:
            coords: [B, 2, H1, W1]
            dilation: optional [B, 1, H1, W1]
        Returns:
            sampled correlation [B, C_total, H1, W1]
        """
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)  # [B,H1,W1,2]
        batch, h1, w1, _ = coords.shape

        if dilation is None:
            dilation = torch.ones(batch, 1, h1, w1, device=coords.device)

        out_pyramid = []

        for level, corr in enumerate(self.corr_pyramid):
            B_corr, C, H2, W2 = corr.shape  # B_corr = batch*h1*w1 for flattened

            # Prepare coords for current pyramid level
            coords_lvl = coords.reshape(batch*h1*w1, 1, 1, 2)  # [B_corr,1,1,2]

            # Rescale coords if correlation map is smaller
            if H2 != h1 or W2 != w1:
                coords_lvl = F.interpolate(
                    coords_lvl.permute(0,3,1,2), size=(H2,W2),
                    mode='bilinear', align_corners=False
                ).permute(0,2,3,1)  # [B_corr,H2,W2,2]

            # Downsample dilation the same way
            dilation_lvl = F.interpolate(
                dilation, size=(H2,W2), mode='bilinear', align_corners=False
            ).reshape(batch*h1*w1, 1, 1, 1)

            # Create delta offsets for local neighborhood
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), dim=-1)  # [2r+1,2r+1,2]
            delta = delta.view(1, 2*r+1, 2*r+1, 2).to(coords.device)

            # Expand coords to add delta safely
            coords_lvl = coords_lvl.unsqueeze(3).unsqueeze(4)  # [B_corr,H2,W2,1,1,2]
            delta = delta.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,2r+1,2r+1,2]

            # Apply delta * dilation
            coords_lvl = coords_lvl + delta * dilation_lvl.unsqueeze(-1)  # broadcast
            coords_lvl = coords_lvl.view(-1, 2*r+1, 2*r+1, 2)  # flatten for sampling

            # Sample correlation
            corr_sampled = bilinear_sampler(corr, coords_lvl)  # [B_corr,1,2r+1,2r+1]

            # reshape back to [B,H1,W1,C]
            corr_sampled = corr_sampled.view(batch, h1, w1, -1)
            out_pyramid.append(corr_sampled)

        # Concatenate all pyramid levels
        out = torch.cat(out_pyramid, dim=-1)
        out = out.permute(0, 3, 1, 2).contiguous().float()  # [B,C_total,H1,W1]

        return out

    @staticmethod
    def corr(fmap1, fmap2, num_head):
        batch, dim, h1, w1 = fmap1.shape
        h2, w2 = fmap2.shape[2:]
        fmap1 = fmap1.view(batch, num_head, dim // num_head, h1 * w1)
        fmap2 = fmap2.view(batch, num_head, dim // num_head, h2 * w2)
        corr = fmap1.transpose(2, 3) @ fmap2
        corr = corr.reshape(batch, num_head, h1, w1, h2, w2).permute(0, 2, 3, 1, 4, 5)
        return corr / torch.sqrt(torch.tensor(dim).float())