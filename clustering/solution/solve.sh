#!/bin/bash
set -e

echo "Clustering"

cd /app


python <<'PYTHON'
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.transform.OnehotEncoder import one_hot_encoder
from src.transform.StandardScaler import standard_scaler

from src.cluster.kmean import kmeans
from src.cluster.spectral import spectral

SEED = 42
DEVICE = "cpu"
BATCH_SIZE = 64
EPOCHS = 600
LEARNING_RATE = 1e-3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class VAE(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.CELU(),
            nn.Linear(256, 128),
            nn.CELU(),
            nn.Linear(128, 64),
            nn.CELU()
        )

        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.CELU(),
            nn.Linear(64, 128),
            nn.CELU(),
            nn.Linear(128, 256),
            nn.CELU(),
            nn.Linear(256, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def vae_loss(x_hat, x, mu, logvar):
    recon = F.mse_loss(x_hat, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl


def run_dataset(
    input_file,
    output_file,
    latent_dim,
    categorical_cols,
    numeric_cols,
    cluster_method
):

    print("=" * 60)
    print("Processing:", input_file)

    df = pd.read_csv(input_file)

    X_onehot = one_hot_encoder(
        df[categorical_cols],
        columns=categorical_cols
    )

    X_scaled = standard_scaler(
        df[numeric_cols]
    )

    X = np.concatenate(
        [X_onehot.values, X_scaled.values],
        axis=1
    ).astype(np.float32)

    input_dim = X.shape[1]

    X_tensor = torch.tensor(X)

    dataset = TensorDataset(X_tensor)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    model = VAE(
        input_dim=input_dim,
        latent_dim=latent_dim
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for (batch,) in loader:

            batch = batch.to(DEVICE)

            optimizer.zero_grad()

            x_hat, mu, logvar = model(batch)

            loss = vae_loss(
                x_hat,
                batch,
                mu,
                logvar
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {
                k: v.clone()
                for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        mu, _ = model.encode(X_tensor.to(DEVICE))

    latent = pd.DataFrame(
        mu.cpu().numpy(),
        columns=[f"z{i+1}" for i in range(latent_dim)]
    )

    if cluster_method == "spectral":
        labels = spectral(latent, k=2)
    else:
        labels = kmeans(latent, k=2)

    pd.DataFrame({
        "pseudo_label": labels.astype(int)
    }).to_csv(
        output_file,
        index=False
    )

    print("Saved:", output_file)


set_seed(SEED)

run_dataset(
    input_file="diabetes_data.csv",
    output_file="output_diabetes_data.csv",
    latent_dim=8,
    categorical_cols=[
        'gender',
        'polyuria',
        'polydipsia',
        'sudden_weight_loss',
        'weakness',
        'polyphagia',
        'genital_thrush',
        'visual_blurring',
        'itching',
        'irritability',
        'delayed_healing',
        'partial_paresis',
        'muscle_stiffness',
        'alopecia',
        'obesity'
    ],
    numeric_cols=[
        'age'
    ],
    cluster_method="spectral"
)

run_dataset(
    input_file="heart.csv",
    output_file="output_heart.csv",
    latent_dim=16,
    categorical_cols=[
        'sex',
        'cp',
        'fbs',
        'restecg',
        'exang',
        'slope',
        'ca',
        'thal'
    ],
    numeric_cols=[
        'trestbps',
        'chol',
        'thalach',
        'oldpeak'
    ],
    cluster_method="kmeans"
)

print("Finished.")

PYTHON