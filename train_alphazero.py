"""Main AlphaZero training script for JungleChess.

Usage:
    python train_alphazero.py
    python train_alphazero.py --resume models/iter_050.pt
"""

import argparse
import copy
import os
import sys
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from alphazero.config import (
    NUM_ITERATIONS, SELF_PLAY_GAMES, TRAINING_STEPS,
    EVAL_GAMES, WIN_RATE_THRESHOLD, CHECKPOINT_DIR, LOG_DIR
)
from alphazero.network import JungleChessNet
from alphazero.self_play import generate_self_play_data
from alphazero.trainer import Trainer
from alphazero.evaluator import evaluate, evaluate_vs_minimax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None, help='cuda / cpu (auto-detect if omitted)')
    args = parser.parse_args()

    device_str = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    print(f"Using device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(LOG_DIR)

    # Initialize networks
    best_net = JungleChessNet().to(device)
    new_net = JungleChessNet().to(device)
    trainer = Trainer(new_net, device)

    start_iter = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        best_net.load_state_dict(ckpt['model'])
        new_net.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            trainer.optimizer.load_state_dict(ckpt['optimizer'])
        start_iter = ckpt.get('iteration', 0) + 1
        print(f"Resumed from {args.resume} at iteration {start_iter}")

    for iteration in range(start_iter, NUM_ITERATIONS):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}")
        print(f"{'='*60}")

        # 1. Self-play
        t0 = time.time()
        print(f"Generating {SELF_PLAY_GAMES} self-play games...")
        new_net.eval()
        samples = generate_self_play_data(new_net, SELF_PLAY_GAMES, device_str)
        trainer.replay_buffer.add(samples)
        print(f"  {len(samples)} samples added | buffer size: {len(trainer.replay_buffer)} | {time.time()-t0:.1f}s")
        writer.add_scalar('self_play/samples_per_iter', len(samples), iteration)
        writer.add_scalar('self_play/buffer_size', len(trainer.replay_buffer), iteration)

        # 2. Train
        t0 = time.time()
        print(f"Training for {TRAINING_STEPS} steps...")
        losses = trainer.train_epoch(TRAINING_STEPS)
        if losses:
            print(f"  policy_loss={losses['policy_loss']:.4f}  value_loss={losses['value_loss']:.4f} | {time.time()-t0:.1f}s")
            writer.add_scalar('train/policy_loss', losses['policy_loss'], iteration)
            writer.add_scalar('train/value_loss', losses['value_loss'], iteration)

        # 3. Evaluate new_net vs best_net
        print(f"Evaluating ({EVAL_GAMES} games)...")
        win_rate = evaluate(new_net, best_net, device_str, EVAL_GAMES)
        writer.add_scalar('eval/win_rate_vs_best', win_rate, iteration)

        if win_rate >= WIN_RATE_THRESHOLD:
            print(f"  New model accepted (win_rate={win_rate:.3f} >= {WIN_RATE_THRESHOLD})")
            best_net.load_state_dict(copy.deepcopy(new_net.state_dict()))
        else:
            print(f"  New model rejected (win_rate={win_rate:.3f} < {WIN_RATE_THRESHOLD}), keeping best")
            new_net.load_state_dict(copy.deepcopy(best_net.state_dict()))
            trainer.network = new_net

        # 4. Periodic minimax evaluation
        if (iteration + 1) % 10 == 0:
            print("Evaluating vs minimax...")
            wr4 = evaluate_vs_minimax(best_net, device_str, minimax_depth=4, num_games=20)
            writer.add_scalar('eval/win_rate_vs_minimax4', wr4, iteration)

        # 5. Save checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"iter_{iteration+1:03d}.pt")
        torch.save({
            'iteration': iteration,
            'model': best_net.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
        }, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

    writer.close()
    print("\nTraining complete.")


if __name__ == '__main__':
    main()
