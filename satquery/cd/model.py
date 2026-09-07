"""A compact Siamese U-Net for bi-temporal change detection.

**Written here rather than imported, for a licence reason.** TinyCD -- the
obvious off-the-shelf choice at F1 91.05 on LEVIR-CD -- states "code is
released for non-commercial and research purposes only", which is the same
restriction that removed VRSBench, OSCD's labels and HRSCD's annotations from
this project. Open-CD is Apache 2.0 but pulls the whole mmcv/mmsegmentation
stack, and its published weights are trained on LEVIR-CD (academic-only) or
S2Looking (no stated licence anywhere). Roughly two hundred lines of our own
code avoids all of it.

**Siamese, not stacked.** The two dates share one encoder, so the network
learns what a building looks like once rather than twice, and the change
signal is the *difference* between two embeddings of the same feature space.
Feeding six channels into a plain U-Net would let it learn date-specific
shortcuts -- the two mosaics differ in sun angle and season, and a model can
score well by detecting "this is the later image" instead of detecting change.

**Initialised from ImageNet, deliberately not from a change-detection
checkpoint.** torchvision's ResNet-18 weights are BSD-3, unencumbered, and
give the encoder edges and texture for free. Every public CD checkpoint traces
back to a restricted dataset.
"""

from __future__ import annotations

__all__ = ["BuildingUNet", "SiameseUNet", "dice_bce_loss", "change_metrics"]


def _conv_block(torch_nn, in_ch: int, out_ch: int):
    return torch_nn.Sequential(
        torch_nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        torch_nn.BatchNorm2d(out_ch),
        torch_nn.ReLU(inplace=True),
        torch_nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        torch_nn.BatchNorm2d(out_ch),
        torch_nn.ReLU(inplace=True),
    )


class SiameseUNet:
    """Constructed lazily so importing this module needs no torch."""

    def __new__(cls, pretrained: bool = True, base: int = 64):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torchvision.models import ResNet18_Weights, resnet18

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                encoder = resnet18(weights=weights)
                # Four stages, kept as separate attributes so the decoder can
                # take skip connections from each resolution.
                self.stem = nn.Sequential(
                    encoder.conv1, encoder.bn1, encoder.relu
                )  # /2, 64
                self.pool = encoder.maxpool  # /4
                self.layer1 = encoder.layer1  # /4,  64
                self.layer2 = encoder.layer2  # /8,  128
                self.layer3 = encoder.layer3  # /16, 256
                self.layer4 = encoder.layer4  # /32, 512

                self.up4 = _conv_block(nn, 512 + 256, 256)
                self.up3 = _conv_block(nn, 256 + 128, 128)
                self.up2 = _conv_block(nn, 128 + 64, 64)
                self.up1 = _conv_block(nn, 64 + 64, base)
                self.head = nn.Conv2d(base, 1, 1)

            def _encode(self, x):
                s0 = self.stem(x)
                s1 = self.layer1(self.pool(s0))
                s2 = self.layer2(s1)
                s3 = self.layer3(s2)
                s4 = self.layer4(s3)
                return s0, s1, s2, s3, s4

            def forward(self, pixels):
                # pixels: (B, 6, H, W) -- two RGB dates stacked on the channel
                # axis by the dataset, split here so both go through the SAME
                # encoder weights.
                a, b = pixels[:, :3], pixels[:, 3:]
                fa = self._encode(a)
                fb = self._encode(b)
                # Absolute difference at every scale. Absolute and not signed:
                # a building appearing and one being demolished are both
                # change, and the mask does not distinguish them.
                d0, d1, d2, d3, d4 = (
                    torch.abs(x - y) for x, y in zip(fa, fb, strict=True)
                )

                x = F.interpolate(d4, size=d3.shape[-2:], mode="bilinear", align_corners=False)
                x = self.up4(torch.cat([x, d3], 1))
                x = F.interpolate(x, size=d2.shape[-2:], mode="bilinear", align_corners=False)
                x = self.up3(torch.cat([x, d2], 1))
                x = F.interpolate(x, size=d1.shape[-2:], mode="bilinear", align_corners=False)
                x = self.up2(torch.cat([x, d1], 1))
                x = F.interpolate(x, size=d0.shape[-2:], mode="bilinear", align_corners=False)
                x = self.up1(torch.cat([x, d0], 1))
                x = F.interpolate(
                    x, size=pixels.shape[-2:], mode="bilinear", align_corners=False
                )
                return self.head(x).squeeze(1)

        return Module()


class BuildingUNet:
    """The same U-Net, one date in, buildings out.

    Shares every layer with SiameseUNet except the input: there is no second
    image and no difference step, so the encoder features go straight to the
    decoder. Kept as a separate class rather than a flag because the two solve
    different problems and conflating them is what produced a model that
    plateaued at F1 0.29.
    """

    def __new__(cls, pretrained: bool = True, base: int = 64):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torchvision.models import ResNet18_Weights, resnet18

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                encoder = resnet18(weights=weights)
                self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
                self.pool = encoder.maxpool
                self.layer1 = encoder.layer1
                self.layer2 = encoder.layer2
                self.layer3 = encoder.layer3
                self.layer4 = encoder.layer4
                self.up4 = _conv_block(nn, 512 + 256, 256)
                self.up3 = _conv_block(nn, 256 + 128, 128)
                self.up2 = _conv_block(nn, 128 + 64, 64)
                self.up1 = _conv_block(nn, 64 + 64, base)
                self.head = nn.Conv2d(base, 1, 1)

            def forward(self, pixels):
                s0 = self.stem(pixels)
                s1 = self.layer1(self.pool(s0))
                s2 = self.layer2(s1)
                s3 = self.layer3(s2)
                s4 = self.layer4(s3)
                x = F.interpolate(s4, size=s3.shape[-2:], mode="bilinear", align_corners=False)
                x = self.up4(torch.cat([x, s3], 1))
                x = F.interpolate(x, size=s2.shape[-2:], mode="bilinear", align_corners=False)
                x = self.up3(torch.cat([x, s2], 1))
                x = F.interpolate(x, size=s1.shape[-2:], mode="bilinear", align_corners=False)
                x = self.up2(torch.cat([x, s1], 1))
                x = F.interpolate(x, size=s0.shape[-2:], mode="bilinear", align_corners=False)
                x = self.up1(torch.cat([x, s0], 1))
                x = F.interpolate(x, size=pixels.shape[-2:], mode="bilinear", align_corners=False)
                return self.head(x).squeeze(1)

        return Module()


def dice_bce_loss(logits, target, bce_weight: float = 0.5, valid=None):
    """BCE plus soft Dice.

    Dice alone is unstable when a crop contains no change at all -- the
    denominator goes to zero and the gradient with it. BCE alone optimises
    pixel accuracy, which a change mask can reach 99% of by predicting nothing,
    since changed pixels are a tiny minority. The pair is the standard answer
    and it is the standard answer for exactly this reason.
    """
    import torch
    import torch.nn.functional as F

    # `valid` zeroes the contribution of pixels no date actually imaged. Left
    # in, a masked region is a free win for predicting change, because the
    # label says "building appeared" and the pixels say nothing at all.
    if valid is None:
        valid = torch.ones_like(target)
    per_pixel = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    denominator = valid.sum().clamp(min=1.0)
    bce = (per_pixel * valid).sum() / denominator
    probs = torch.sigmoid(logits) * valid
    target = target * valid
    intersection = (probs * target).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = 1 - (2 * intersection + 1.0) / (union + 1.0)
    return bce_weight * bce + (1 - bce_weight) * dice.mean()


def change_metrics(logits, target, threshold: float = 0.5, valid=None) -> dict[str, float]:
    """F1 and IoU over the changed class only.

    Over the changed class and not averaged with background: a mask that is 98%
    background scores 98% "accuracy" while finding nothing, which is the number
    that makes a useless model look finished.
    """
    import torch

    with torch.no_grad():
        if valid is None:
            valid = torch.ones_like(target)
        predicted = (torch.sigmoid(logits) > threshold).float() * valid
        target = target * valid
        tp = (predicted * target).sum()
        fp = (predicted * (1 - target)).sum()
        fn = ((1 - predicted) * target).sum()
        f1 = (2 * tp / (2 * tp + fp + fn)).item() if (2 * tp + fp + fn) > 0 else 0.0
        iou = (tp / (tp + fp + fn)).item() if (tp + fp + fn) > 0 else 0.0
        return {
            "f1": f1,
            "iou": iou,
            "positive_share": target.mean().item(),
            "predicted_share": predicted.mean().item(),
        }
