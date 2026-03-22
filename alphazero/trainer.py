"""Replay buffer and training loop for AlphaZero."""

import random
import collections
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

from alphazero.config import (
    REPLAY_BUFFER_SIZE, BATCH_SIZE, LEARNING_RATE,
    LR_MILESTONES, LR_GAMMA, WEIGHT_DECAY, TRAINING_STEPS, ACTION_SPACE
)


class ReplayBuffer:
    """Circular buffer storing (state, policy, value) tuples."""

    def __init__(self, maxlen=REPLAY_BUFFER_SIZE):
        self.buffer = collections.deque(maxlen=maxlen)

    def add(self, samples):
        """Add a list of (state_planes, policy, outcome) tuples."""
        self.buffer.extend(samples)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, policies, values = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(policies)),
            torch.FloatTensor(np.array(values)),
        )

    def __len__(self):
        return len(self.buffer)


class Trainer:
    def __init__(self, network, device):
        self.network = network
        self.device = device
        self.optimizer = torch.optim.Adam(
            network.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        self.scheduler = MultiStepLR(
            self.optimizer,
            milestones=LR_MILESTONES,
            gamma=LR_GAMMA,
        )
        self.replay_buffer = ReplayBuffer()

    def train_step(self, states, policies, values):
        """Single gradient update. Returns (policy_loss, value_loss)."""
        self.network.train()
        states = states.to(self.device)
        policies = policies.to(self.device)
        values = values.to(self.device)

        logits, pred_values = self.network(states)

        # Policy loss: cross-entropy against MCTS visit counts
        policy_loss = -(policies * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

        # Value loss: MSE
        value_loss = F.mse_loss(pred_values, values)

        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        return policy_loss.item(), value_loss.item()

    def train_epoch(self, steps=TRAINING_STEPS):
        """Run `steps` gradient updates from the replay buffer.

        Returns dict of average losses.
        """
        if len(self.replay_buffer) < BATCH_SIZE:
            return {}

        total_p_loss = 0.0
        total_v_loss = 0.0
        for _ in range(steps):
            states, policies, values = self.replay_buffer.sample(BATCH_SIZE)
            p_loss, v_loss = self.train_step(states, policies, values)
            total_p_loss += p_loss
            total_v_loss += v_loss

        self.scheduler.step()

        return {
            'policy_loss': total_p_loss / steps,
            'value_loss': total_v_loss / steps,
        }
