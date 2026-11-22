from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _load_split(root, split):
    files = sorted((Path(root) / split).glob("*.txt"))
    return [pd.read_table(path, sep="\t") for path in files]


def load_synergy_data(data_path):
    return (
        _load_split(data_path, "train"),
        _load_split(data_path, "val"),
        _load_split(data_path, "test"),
    )


class MiniCellDataSet:
    def __init__(
        self,
        batch_size,
        samples_support=10,
        samples_query=15,
        data_path="./data/drugcomb_few_shot",
        seed=1500,
        val_cell_rate=0.2,
    ):
        train, val, test = load_synergy_data(data_path)
        if not train:
            raise FileNotFoundError(
                f"No training cell files found under {data_path}/train"
            )
        if not test:
            raise FileNotFoundError(f"No test cell files found under {data_path}/test")

        train_rng = np.random.default_rng(seed)
        if not val:
            order = train_rng.permutation(len(train))
            val_count = max(1, int(round(len(train) * val_cell_rate)))
            val_ids = set(order[:val_count].tolist())
            val = [task for idx, task in enumerate(train) if idx in val_ids]
            train = [task for idx, task in enumerate(train) if idx not in val_ids]

        self.batch_size = batch_size
        self.samples_support = samples_support
        self.samples_query = samples_query
        self.datasets = {"train": train, "val": val, "test": test}
        val_seed, test_seed = np.random.SeedSequence(seed).spawn(2)
        self.rngs = {
            "train": train_rng,
            "val": np.random.default_rng(val_seed),
            "test": np.random.default_rng(test_seed),
        }
        required = samples_support + samples_query
        for split, tasks in self.datasets.items():
            too_small = [len(task) for task in tasks if len(task) < required]
            if too_small:
                raise ValueError(
                    f"{split} contains a cell with fewer than {required} samples"
                )

    def __len__(self):
        return len(self.datasets["train"])

    def _sample_batch(self, split):
        tasks = self.datasets[split]
        rng = self.rngs[split]
        support = np.empty(
            (self.batch_size, self.samples_support, 4), dtype=np.float64
        )
        query = np.empty(
            (self.batch_size, self.samples_query, 4), dtype=np.float64
        )
        for batch_idx in range(self.batch_size):
            task = tasks[rng.integers(0, len(tasks))].to_numpy()
            ids = rng.choice(
                len(task), self.samples_support + self.samples_query, replace=False
            )
            episode = task[ids].astype(np.float64, copy=False)
            support[batch_idx] = episode[: self.samples_support]
            query[batch_idx] = episode[self.samples_support :]
        return torch.from_numpy(support), torch.from_numpy(query)

    def get_train_batch(self):
        return self._sample_batch("train")

    def get_val_batch(self):
        return self._sample_batch("val")

    def get_test_batch(self):
        return self._sample_batch("test")


class RepresentationData:
    def __init__(self, data_path="./data/drugcomb_few_shot", val_rate=0.1, seed=1500):
        train_tasks = _load_split(data_path, "train")
        if not train_tasks:
            raise FileNotFoundError(
                f"No training cell files found under {data_path}/train"
            )
        rng = np.random.default_rng(seed)
        train_rows, val_rows = [], []
        for task in train_tasks:
            values = task.to_numpy()
            order = rng.permutation(len(values))
            val_count = max(1, int(round(len(values) * val_rate)))
            val_rows.append(values[order[:val_count]])
            train_rows.append(values[order[val_count:]])
        self.train = np.concatenate(train_rows).astype(np.float64)
        self.val = np.concatenate(val_rows).astype(np.float64)
