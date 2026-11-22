import numpy as np
import torch
from scipy import stats
from sklearn.metrics import r2_score
from torch_geometric import data as geometric_data
from torch_geometric.data import Dataset

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader


def _mapping_index(mapping, value):
    candidates = (value, int(value), str(int(value)))
    for candidate in candidates:
        index = mapping.item2idx.get(candidate)
        if index is not None:
            return index
    raise KeyError(f"Feature ID not found: {value}")


class TestbedDataset(Dataset):
    def __init__(self, rows, drug_features, cell_features, codes):
        super().__init__()
        self.rows = rows
        self.drug_features = drug_features
        self.cell_features = cell_features
        self.codes = codes

    def len(self):
        return len(self.rows)

    def get(self, idx):
        cell_idx = _mapping_index(self.codes["cells"], self.rows[idx, 0])
        drug_idx = _mapping_index(self.codes["drugs"], self.rows[idx, 1])
        drug = self.drug_features.iloc[drug_idx]
        cell = self.cell_features.iloc[cell_idx]
        graph = geometric_data.Data(
            x=drug["features"], edge_index=drug["edge_index"].transpose(1, 0)
        )
        graph.c = cell["cg"].unsqueeze(0)
        graph.y = torch.tensor([float(self.rows[idx, 3])], dtype=torch.float32)
        graph.c_size1 = drug["c_size"].long()
        return graph


class TestbedDataset1(Dataset):
    def __init__(self, rows, drug_features, cell_features, codes):
        super().__init__()
        self.rows = rows
        self.drug_features = drug_features
        self.codes = codes

    def len(self):
        return len(self.rows)

    def get(self, idx):
        drug_idx = _mapping_index(self.codes["drugs"], self.rows[idx, 2])
        drug = self.drug_features.iloc[drug_idx]
        graph = geometric_data.Data(
            x=drug["features"], edge_index=drug["edge_index"].transpose(1, 0)
        )
        graph.c_size2 = drug["c_size"].long()
        return graph


def paired_loaders(
    rows, drug_features, cell_features, codes, batch_size, shuffle=False
):
    if shuffle:
        rows = rows[np.random.permutation(len(rows))]
    first = TestbedDataset(rows, drug_features, cell_features, codes)
    second = TestbedDataset1(rows, drug_features, cell_features, codes)
    return (
        DataLoader(first, batch_size=batch_size, shuffle=False),
        DataLoader(second, batch_size=batch_size, shuffle=False),
    )


def mse(y, pred):
    return float(np.mean((y - pred) ** 2))


def rmse(y, pred):
    return float(np.sqrt(mse(y, pred)))


def pearson(y, pred):
    return float(np.corrcoef(y, pred)[0, 1])


def spearman(y, pred):
    return float(stats.spearmanr(y, pred)[0])


def metrics(y, pred):
    return {
        "mse": mse(y, pred),
        "rmse": rmse(y, pred),
        "spearman": spearman(y, pred),
        "pearson": pearson(y, pred),
        "r2": float(r2_score(y, pred)),
    }
