import argparse
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
    drug_a = next(iter(loader_a)).to(device)
    drug_b = next(iter(loader_b)).to(device)
    return drug_a, drug_b


def evaluate(model, sampler, features, device, args):
    model.eval()
    labels, predictions = [], []
    for _ in range(args.val_batches):
        support, query = sampler.get_val_batch()
        drug_a, drug_b = prepare_episode(support, query, features, device)
        prediction, target = model.predict_batch(
            drug_a, drug_b, args.batch_size, args.adaptation_steps, args.adaptation_lr
        )
        predictions.append(prediction.cpu().flatten())
        labels.append(target.cpu().flatten())
    return metrics(torch.cat(labels).numpy(), torch.cat(predictions).numpy())


def main():
    parser = argparse.ArgumentParser(description="Meta-train DiffPro")
    parser.add_argument("--data-path", default="data/drugcomb_few_shot")
    parser.add_argument("--feature-path", default="data/sample_features")
    parser.add_argument(
        "--pretrained",
        default="checkpoints/diffpro_encoder.pt",
    )
    parser.add_argument("--output", default="checkpoints/diffpro.pt")
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--support", type=int, default=5)
    parser.add_argument("--query", type=int, default=40)
    parser.add_argument("--val-batches", type=int, default=30)
    parser.add_argument("--val-interval", type=int, default=2000)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--diff-weight", type=float, default=0.05)
    parser.add_argument("--adaptation-steps", type=int, default=15)
    parser.add_argument("--adaptation-lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1500)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.support < 1 or args.query < 1 or args.diffusion_steps < 1:
        parser.error("support, query and diffusion steps must be positive")
    if args.adaptation_steps < 0 or args.adaptation_lr <= 0 or args.diff_weight < 0:
        parser.error("adaptation steps and diffusion weight must be nonnegative, and adaptation LR positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    features = load_features(resolve(args.feature_path))
    sampler = MiniCellDataSet(
        batch_size=args.batch_size,
        samples_support=args.support,
        samples_query=args.query,
        data_path=resolve(args.data_path),
        seed=args.seed,
    )

    model = DiffPro(args.support, args.query, args.diffusion_steps).to(device)
    pretrained = resolve(args.pretrained)
    if not pretrained.is_file():
        raise FileNotFoundError(
            f"Missing pretrained extractor: {pretrained}. "
            "Run pretrain_diffpro.py first."
        )
    model.extractor.load_state_dict(torch.load(pretrained, map_location=device))
    model.extractor.requires_grad_(False)

    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
    )

    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_mse = float("inf")
    for step in range(1, args.steps + 1):
        model.train()
        model.extractor.eval()
        support, query = sampler.get_train_batch()
        drug_a, drug_b = prepare_episode(support, query, features, device)
        optimizer.zero_grad()
        total_loss, query_loss, diffusion_loss = model.meta_train_batch(
            drug_a, drug_b, args.batch_size, args.diff_weight
        )
        total_loss.backward()
        optimizer.step()

        if step == 1 or step % args.log_interval == 0:
            print(
                f"step={step:04d} total={total_loss.item():.5f} "
                f"query={query_loss.item():.5f} diffusion={diffusion_loss.item():.5f}"
            )
        if step % args.val_interval == 0 or step == args.steps:
            result = evaluate(model, sampler, features, device, args)
            print(
                f"validation step={step:04d} mse={result['mse']:.5f} "
                f"spearman={result['spearman']:.4f} r2={result['r2']:.4f}"
            )
            if result["mse"] < best_mse:
                best_mse = result["mse"]
                torch.save(
                    {
                        "step": step,
                        "support": args.support,
                        "query": args.query,
                        "diffusion_steps": args.diffusion_steps,
                        "diff_weight": args.diff_weight,
                        "adaptation_steps": args.adaptation_steps,
                        "adaptation_lr": args.adaptation_lr,
                        "seed": args.seed,
                        "state_dict": model.state_dict(),
                        "validation": result,
                    },
                    output,
                )
                print(f"saved best model to {output}")


if __name__ == "__main__":
    main()
