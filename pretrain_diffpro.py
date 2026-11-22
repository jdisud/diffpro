import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset import RepresentationData
from pretrain_model import EMBNet
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


def run_epoch(model, rows, features, batch_size, device, optimizer=None):
    codes, drugs, cells = features
    if optimizer is not None:
        order = np.random.permutation(len(rows))
        rows = rows[order]
        model.train()
    else:
        model.eval()

    loader_a, loader_b = paired_loaders(
        rows, drugs, cells, codes, batch_size=batch_size, shuffle=False
    )
    labels, predictions = [], []
    for drug_a, drug_b in zip(loader_a, loader_b):
        drug_a, drug_b = drug_a.to(device), drug_b.to(device)
        output = model(drug_a, drug_b)
        loss = F.mse_loss(output, drug_a.y.view(-1, 1))
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        labels.append(drug_a.y.detach().cpu().view(-1))
        predictions.append(output.detach().cpu().view(-1))
    return metrics(torch.cat(labels).numpy(), torch.cat(predictions).numpy())


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain the shared DiffPro encoder"
    )
    parser.add_argument("--data-path", default="data/drugcomb_few_shot")
    parser.add_argument("--feature-path", default="data/sample_features")
    parser.add_argument(
        "--output",
        default="checkpoints/diffpro_encoder.pt",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=1500)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    data = RepresentationData(resolve(args.data_path), seed=args.seed)
    features = load_features(resolve(args.feature_path))
    model = EMBNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_mse = float("inf")
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(
            model, data.train, features, args.batch_size, device, optimizer
        )
        with torch.no_grad():
            val_result = run_epoch(model, data.val, features, args.batch_size, device)
        print(
            f"epoch={epoch:03d} train_mse={train_result['mse']:.5f} "
            f"val_mse={val_result['mse']:.5f} val_spearman={val_result['spearman']:.4f}"
        )
        if val_result["mse"] < best_mse:
            best_mse = val_result["mse"]
            torch.save(model.extractor.state_dict(), output)
    print(f"saved best feature extractor to {output}")


if __name__ == "__main__":
    main()
