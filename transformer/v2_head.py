import torch
import torch.nn as nn
import torch.nn.functional as F

# --- same blocks as PredTrAD_v2 (DoubleConvolution + Down) ---
class DoubleConvolution(nn.Module):
    """(conv3x3 -> BN -> ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class Down(nn.Module):
    """MaxPool(2x2) then DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            DoubleConvolution(in_ch, out_ch),
            nn.MaxPool2d(2)
        )
    def forward(self, x): return self.block(x)

# --- PredTrAD-V2 style CNN pyramid head for classification ---
class PredTrADV2Head(nn.Module):
    """
    Input: encoder features H of shape (B, L, d_model)
    Steps: reshape to (B,1,L,d_model) -> Down x3 -> 1x1 conv -> GAP -> Linear -> logit
    """
    def __init__(self, out_ch_after_1x1: int = 64, dropout: float = 0.10):
        super().__init__()
        self.down1 = Down(1,   64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128,256)
        # 1x1 conv like in PredTrAD_v2 (they go 256->1 then do a linear).
        # For classification, keep some channels, then global-average-pool.
        self.proj1x1 = nn.Conv2d(256, out_ch_after_1x1, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch_after_1x1)
        self.act = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)  # (B,C,1,1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(out_ch_after_1x1, 1)

    def _pad_to_mult8(self, x):
        # make H and W divisible by 8 so that 3x MaxPool(2) is safe
        _, _, H, W = x.shape
        pad_h = (8 - (H % 8)) % 8
        pad_w = (8 - (W % 8)) % 8
        return F.pad(x, (0, pad_w, 0, pad_h)) if (pad_h or pad_w) else x

    def forward(self, H):              # H: (B, L, d_model)
        x = H.unsqueeze(1)             # (B,1,L,d_model)  treat as image (sensors x time)
        x = self._pad_to_mult8(x)
        x = self.down1(x)              # -> (B,  64, L/2, d_model/2)
        x = self.down2(x)              # -> (B, 128, L/4, d_model/4)
        x = self.down3(x)              # -> (B, 256, L/8, d_model/8)
        x = self.proj1x1(x)            # -> (B, out_ch, L/8, d_model/8)
        x = self.bn(x); x = self.act(x)
        x = self.gap(x).flatten(1)     # -> (B, out_ch)
        x = self.drop(x)
        return self.fc(x).squeeze(-1)  # -> (B,)

class FaultClassifierV2(nn.Module):
    """
    Wraps your TransformerAutoencoder backbone and applies PredTrADV2Head.
    """
    def __init__(self, backbone, dropout: float = 0.10):
        super().__init__()
        self.backbone = backbone
        self.head = PredTrADV2Head(dropout=dropout)

    def forward(self, src):            # src: (B, L, D_in)
        H = self.backbone.encode(src)  # (B, L, d_model)
        return self.head(H)            # (B,)
