import argparse
import csv
import pickle
import random
from pathlib import Path

import numpy as np
import torch

from dataset import MiniCellDataSet
from diffpro import DiffPro
from data_utils import metrics, paired_loaders


ROOT = Path(__file__).resolve().parent


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_features(feature_dir):
    with open(feature_dir / "codes_cell.p", "rb") as handle:
        codes = pickle.load(handle)
    with open(feature_dir / "drug_feature_cell.p", "rb") as handle:
        drugs = pickle.load(handle)
    with open(feature_dir / "cell_feature_900.p", "rb") as handle:
        cells = pickle.load(handle)
    return codes, drugs, cells


def prepare_episode(support, query, features, device):
    rows = torch.cat((support, query), dim=1).reshape(-1, 4).numpy()
    codes, drugs, cells = features
    loader_a, loader_b = paired_loaders(
        rows, drugs, cells, codes, batch_size=len(rows), shuffle=False
    )
    return next(iter(loader_a)).to(device), next(iter(loader_b)).to(device)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DiffPro model"
    )
    parser.add_argument("--checkpoint", default="checkpoints/diffpro.pt")
    parser.add_argument("--data-path", default="data/drugcomb_few_shot")
    parser.add_argument("--feature-path", default="data/sample_features")
    parser.add_argument("--output")
    parser.add_argument("--support", type=int, default=5)
    parser.add_argument("--query", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--test-batches", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--adaptation-steps", type=int)
    parser.add_argument("--adaptation-lr", type=float)
    parser.add_argument("--seed", type=int, default=1500)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.support < 1 or args.query < 1:
        parser.error("few-shot evaluation requires positive support and query sizes")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint_path = resolve(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}. Run train_diffpro.py first."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "diffusion_steps" not in checkpoint:
        raise ValueError("Legacy checkpoint. Re-run representation learning and DiffPro meta-training.")
    adaptation_steps = args.adaptation_steps
    if adaptation_steps is None:
        adaptation_steps = checkpoint["adaptation_steps"]
    adaptation_lr = args.adaptation_lr
    if adaptation_lr is None:
        adaptation_lr = checkpoint["adaptation_lr"]
    if adaptation_steps < 0 or adaptation_lr <= 0:
        parser.error("adaptation steps must be nonnegative and adaptation LR positive")

    model = DiffPro(args.support, args.query, checkpoint["diffusion_steps"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.requires_grad_(False)
    model.eval()

    features = load_features(resolve(args.feature_path))
    sampler = MiniCellDataSet(
        batch_size=args.batch_size,
        samples_support=args.support,
        samples_query=args.query,
        data_path=resolve(args.data_path),
        seed=args.seed,
    )
    output = resolve(args.output or f"results/diffpro_{args.support}shot.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["round", "mse", "rmse", "spearman", "pearson", "r2"]
    rows = []
    for round_index in range(args.rounds):
        labels, predictions = [], []
        for _ in range(args.test_batches):
            support_rows, query_rows = sampler.get_test_batch()
            drug_a, drug_b = prepare_episode(
                support_rows, query_rows, features, device
            )
            prediction, target = model.predict_batch(
                drug_a, drug_b, args.batch_size, adaptation_steps, adaptation_lr
            )
            predictions.append(prediction.cpu().flatten())
            labels.append(target.cpu().flatten())
        result = metrics(torch.cat(labels).numpy(), torch.cat(predictions).numpy())
        result["round"] = round_index
        rows.append(result)
        print(
            f"round={round_index:02d} mse={result['mse']:.5f} "
            f"spearman={result['spearman']:.4f} r2={result['r2']:.4f}"
        )

    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved test metrics to {output}")


if __name__ == "__main__":
    main()
