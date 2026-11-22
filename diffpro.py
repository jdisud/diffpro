import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import SampleFeatureExtractor


class DiffusionModule(nn.Module):
    def __init__(self, n_features=128, n_steps=50, hidden_dim=128):
        super().__init__()
        self.n_steps = n_steps
        self.net = nn.Sequential(
            nn.Linear(n_features + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )
        betas = torch.linspace(1.0e-4, 2.0e-2, n_steps)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

    def forward(self, x, t):
        return self.net(torch.cat((x, t), dim=1))

    def loss(self, x0):
        batch_size = x0.size(0)
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device)
        noise = torch.randn_like(x0)
        alpha = self.alphas_cumprod[t].view(-1, 1)
        x_noisy = alpha.sqrt() * x0 + (1.0 - alpha).sqrt() * noise
        t_in = t.float().view(-1, 1) / max(self.n_steps - 1, 1)
        error = self.forward(x_noisy, t_in) - noise
        return error.square().sum(dim=-1).mean()


class WeightGenerativeNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.metanet_g1 = nn.Linear(128, 512 * 256 + 256)
        self.metanet_g2 = nn.Linear(128, 256 + 1)

    def forward(self, latents):
        final1 = self.metanet_g1(latents)
        final2 = self.metanet_g2(latents)
        weights1 = F.normalize(final1[:, : 512 * 256], p=2, dim=1)
        bias1 = final1[:, 512 * 256 :]
        weights2 = F.normalize(final2[:, :256], p=2, dim=1)
        bias2 = final2[:, 256:]
        return weights1, bias1, weights2, bias2


class InferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, 3, 1, 1)
        self.norm1 = nn.LayerNorm([64, 16, 16])
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.norm2 = nn.LayerNorm([64, 8, 8])
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 256, 3, 1, 1)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, task_embedding):
        batch_size, shots, length = task_embedding.size()
        side = int((length / 2) ** 0.5)
        if side * side * 2 != length:
            raise ValueError("Task embedding length must be twice a square number")
        x = task_embedding.view(batch_size, shots, 2, side, side)
        x = x.contiguous().view(batch_size * shots, 2, side, side)
        x = self.dropout(x)
        x = self.pool1(F.relu(self.norm1(self.conv1(x))))
        x = self.pool2(F.relu(self.norm2(self.conv2(x))))
        x = self.conv3(x).mean(dim=(2, 3))
        x = x.view(batch_size, shots, -1).mean(dim=1)
        return x.chunk(2, dim=-1)


class DiffPro(nn.Module):
    def __init__(self, num_support, num_query, diffusion_steps=50):
        super().__init__()
        self.extractor = SampleFeatureExtractor()
        self.inference = InferenceModel()
        self.generative = WeightGenerativeNetwork()
        self.diffusion = DiffusionModule(n_steps=diffusion_steps)
        self.num_support = num_support
        self.num_query = num_query
        if num_support < 1 or num_query < 1:
            raise ValueError("Few-shot episodes require positive support and query sizes")

    def encode_episode(self, drug_a, drug_b, batch_size):
        sizes = (self.num_support, self.num_query)
        embeddings = self.extractor(drug_a, drug_b, batch_size)
        labels = drug_a.y.view(batch_size, -1)
        return embeddings.split(sizes, dim=1), labels.split(sizes, dim=1)

    def meta_train_batch(self, drug_a, drug_b, batch_size, diff_weight):
        (support, query), (_, query_labels) = self.encode_episode(
            drug_a, drug_b, batch_size
        )
        mean, log_var = self.inference(support)
        latent = mean + torch.exp(0.5 * log_var) * torch.randn_like(mean)
        predictions = self.synergy_prediction_network(query, self.generative(latent))
        query_loss = F.mse_loss(predictions, query_labels)
        diffusion_loss = self.diffusion.loss(latent)
        return query_loss + diff_weight * diffusion_loss, query_loss, diffusion_loss

    @torch.no_grad()
    def predict_batch(self, drug_a, drug_b, batch_size, adaptation_steps, adaptation_lr):
        (support, query), (support_labels, query_labels) = self.encode_episode(
            drug_a, drug_b, batch_size
        )
        latent, _ = self.inference(support)
        for _ in range(adaptation_steps):
            with torch.enable_grad():
                latent.requires_grad_(True)
                predictions = self.synergy_prediction_network(
                    support, self.generative(latent)
                )
                # Each task has its own support mean and independent update.
                support_loss = (predictions - support_labels).square().mean(dim=1).sum()
                gradient, = torch.autograd.grad(support_loss, latent)
            latent = latent - adaptation_lr * gradient
        predictions = self.synergy_prediction_network(query, self.generative(latent))
        return predictions, query_labels

    @staticmethod
    def synergy_prediction_network(inputs, weights):
        weights1, bias1, weights2, bias2 = weights
        batch_size, samples, _ = inputs.size()
        weights1 = weights1.view(batch_size, 256, 512)
        hidden = torch.bmm(inputs, weights1.transpose(1, 2)) + bias1.unsqueeze(1)
        hidden = F.relu(hidden)
        weights2 = weights2.view(batch_size, 1, 256)
        output = torch.bmm(hidden, weights2.transpose(1, 2)) + bias2.view(
            batch_size, 1, 1
        )
        return output.squeeze(-1).view(batch_size, samples)
