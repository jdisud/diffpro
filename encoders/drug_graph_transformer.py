import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch


class MolecularAttentionLayer(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.bond_bias = nn.Parameter(torch.zeros(()))

    def forward(self, atoms, valid, bonds):
        batch_size, atom_count, dim = atoms.shape
        qkv = self.qkv(atoms).view(batch_size, atom_count, 3, self.heads, dim // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        # Bonds can influence the scores without restricting attention to neighbors.
        bias = (bonds.unsqueeze(1) * self.bond_bias).to(dtype=query.dtype)
        bias = bias.masked_fill(~valid[:, None, None, :], float("-inf"))
        attention = F.scaled_dot_product_attention(
            query, key, value, attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attention = attention.transpose(1, 2).reshape(batch_size, atom_count, dim)
        return self.output(attention) + self.ffn(atoms)


class DrugGraphTransformer(nn.Module):
    def __init__(self, in_dim=78, out_dim=128, heads=4, dropout=0.1):
        super().__init__()
        if out_dim % heads:
            raise ValueError("Drug embedding dimension must be divisible by the number of heads")
        self.atom_projection = nn.Linear(in_dim, out_dim)
        self.layers = nn.ModuleList(
            MolecularAttentionLayer(out_dim, heads, dropout) for _ in range(3)
        )

    def forward(self, x, edge_index, batch):
        atoms, valid = to_dense_batch(self.atom_projection(x), batch)
        bonds = to_dense_adj(
            edge_index, batch, max_num_nodes=atoms.size(1), batch_size=atoms.size(0)
        ).bool()
        for layer in self.layers:
            atoms = layer(atoms, valid, bonds)
        return atoms.masked_fill(~valid.unsqueeze(-1), 0.0).sum(dim=1)
