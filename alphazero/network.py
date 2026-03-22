import torch
import torch.nn as nn
import torch.nn.functional as F

from alphazero.config import (
    INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS,
    NUM_RES_BLOCKS, NUM_FILTERS, POLICY_FILTERS, VALUE_FILTERS, ACTION_SPACE
)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class JungleChessNet(nn.Module):
    """ResNet-based dual-head network for JungleChess AlphaZero.

    Input : (batch, INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS)
    Output: policy logits (batch, ACTION_SPACE), value (batch,)
    """

    def __init__(self):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, NUM_FILTERS, 3, padding=1, bias=False),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.ReLU(inplace=True),
        )
        # Residual tower
        self.res_blocks = nn.Sequential(*[ResBlock(NUM_FILTERS) for _ in range(NUM_RES_BLOCKS)])

        flat = BOARD_ROWS * BOARD_COLS

        # Policy head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, POLICY_FILTERS, 1, bias=False),
            nn.BatchNorm2d(POLICY_FILTERS),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(POLICY_FILTERS * flat, ACTION_SPACE)

        # Value head
        self.value_conv = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, VALUE_FILTERS, 1, bias=False),
            nn.BatchNorm2d(VALUE_FILTERS),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(VALUE_FILTERS * flat, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)

        p = self.policy_conv(x).flatten(1)
        p = self.policy_fc(p)          # raw logits

        v = self.value_conv(x).flatten(1)
        v = self.value_fc(v).squeeze(1)

        return p, v

    @torch.no_grad()
    def predict(self, state_tensor, legal_mask):
        """Single-sample inference with legal move masking.

        Args:
            state_tensor: (INPUT_CHANNELS, H, W) float32 tensor
            legal_mask  : (ACTION_SPACE,) bool tensor

        Returns:
            policy: (ACTION_SPACE,) numpy array (probabilities over legal moves)
            value : float scalar
        """
        self.eval()
        x = state_tensor.unsqueeze(0).to(next(self.parameters()).device)
        logits, v = self(x)
        logits = logits.squeeze(0)

        # Mask illegal moves
        logits[~legal_mask.to(logits.device)] = float('-inf')
        policy = torch.softmax(logits, dim=0).cpu().numpy()
        return policy, v.item()
