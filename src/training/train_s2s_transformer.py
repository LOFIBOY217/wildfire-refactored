"""
Train S2S Transformer for Subseasonal FWI Forecasting
======================================================
Uses historical FWI (encoder) + ECMWF S2S forecasts (decoder) to predict
FWI 14-46 days ahead.

Usage:
    python -m src.training.train_s2s_transformer \\
        --fwi_start 2025-02-09 --fwi_end 2025-12-31 \\
        --config configs/default.yaml

Based on train_s2s_transformer.py.
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "config.py").exists():
            sys.path.insert(0, str(parent))
            break
    from src.config import load_config, get_path, add_config_argument
from src.utils.seed import set_seed
from src.utils.date_utils import parse_date_arg
from src.models.s2s_transformer import S2STransformer
from src.datasets.s2s_fwi import S2SFWIDataset, build_valid_samples


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for encoder_patches, decoder_patches, target_patches in dataloader:
        B, Np = encoder_patches.shape[:2]

        encoder_input = encoder_patches.view(B * Np, *encoder_patches.shape[2:]).to(device)
        decoder_input = decoder_patches.view(B * Np, *decoder_patches.shape[2:]).to(device)
        target = target_patches.view(B * Np, *target_patches.shape[2:]).to(device)

        optimizer.zero_grad()
        output = model(encoder_input, decoder_input)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Validate."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for encoder_patches, decoder_patches, target_patches in dataloader:
            B, Np = encoder_patches.shape[:2]

            encoder_input = encoder_patches.view(B * Np, *encoder_patches.shape[2:]).to(device)
            decoder_input = decoder_patches.view(B * Np, *decoder_patches.shape[2:]).to(device)
            target = target_patches.view(B * Np, *target_patches.shape[2:]).to(device)

            output = model(encoder_input, decoder_input)
            loss = criterion(output, target)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train S2S FWI Transformer")
    add_config_argument(parser)

    # Data args
    parser.add_argument("--fwi_dir", type=str, default=None)
    parser.add_argument("--ecmwf_dir", type=str, default=None)
    parser.add_argument("--fwi_start", type=str, required=True)
    parser.add_argument("--fwi_end", type=str, required=True)
    parser.add_argument("--encoder_days", type=int, default=7)
    parser.add_argument("--lead_start", type=int, default=14)
    parser.add_argument("--lead_end", type=int, default=46)

    # Model args
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training args
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_val_split", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)

    # Output args
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    fwi_dir = args.fwi_dir or get_path(cfg, 'fwi_dir')
    ecmwf_dir = args.ecmwf_dir or get_path(cfg, 'ecmwf_dir')
    output_dir = args.output_dir or os.path.join(get_path(cfg, 'output_dir'), 'outputs_s2s')
    checkpoint_dir = args.checkpoint_dir or get_path(cfg, 'checkpoint_dir')

    set_seed(args.seed)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    fwi_start = parse_date_arg(args.fwi_start)
    fwi_end = parse_date_arg(args.fwi_end)

    # Build samples
    samples = build_valid_samples(
        fwi_dir=fwi_dir, ecmwf_dir=ecmwf_dir,
        fwi_start=fwi_start, fwi_end=fwi_end,
        encoder_days=args.encoder_days,
        lead_start=args.lead_start, lead_end=args.lead_end
    )

    if len(samples) == 0:
        print("Error: no valid samples found!")
        return

    # Train/val split
    split_idx = int(len(samples) * args.train_val_split)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    print(f"\nTrain: {len(train_samples)}, Val: {len(val_samples)}")

    decoder_days = args.lead_end - args.lead_start + 1

    train_dataset = S2SFWIDataset(
        train_samples, fwi_dir, ecmwf_dir,
        args.encoder_days, args.patch_size, args.lead_start, args.lead_end
    )
    val_dataset = S2SFWIDataset(
        val_samples, fwi_dir, ecmwf_dir,
        args.encoder_days, args.patch_size, args.lead_start, args.lead_end
    )

    # Use training set normalization for validation
    val_dataset.fwi_mean = train_dataset.fwi_mean
    val_dataset.fwi_std = train_dataset.fwi_std
    val_dataset.ecmwf_means = train_dataset.ecmwf_means
    val_dataset.ecmwf_stds = train_dataset.ecmwf_stds

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    patch_dim_fwi = args.patch_size * args.patch_size
    patch_dim_ecmwf = args.patch_size * args.patch_size * 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    model = S2STransformer(
        patch_dim_fwi=patch_dim_fwi, patch_dim_ecmwf=patch_dim_ecmwf,
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.d_model * 4, dropout=args.dropout,
        encoder_days=args.encoder_days, decoder_days=decoder_days
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{args.epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, "s2s_transformer_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': args,
                'fwi_mean': train_dataset.fwi_mean,
                'fwi_std': train_dataset.fwi_std,
                'ecmwf_means': train_dataset.ecmwf_means,
                'ecmwf_stds': train_dataset.ecmwf_stds,
                'hw': train_dataset.hw,
                'grid': train_dataset.grid
            }, checkpoint_path)
            print(f"  -> Saved best model (val_loss: {val_loss:.6f})")

    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
