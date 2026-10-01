# Unsupervised Clustering on Diabetes and Heart Disease Datasets

This project implements an **unsupervised clustering pipeline** for two tabular healthcare datasets:

- `diabetes_data.csv`
- `heart.csv`

The ground-truth labels are hidden from the solution. Each dataset contains **two underlying clusters (`k = 2`)**, and the goal is to generate cluster assignments that satisfy the required **Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)** thresholds.

## Task Requirements

The solution must generate the following files:

```text
/app/output_diabetes_data.csv
/app/output_heart.csv
```

Each output file must:

- contain exactly one column named `pseudo_label`;
- contain the same number of rows as the corresponding input dataset;
- use only cluster labels `0` and `1`;
- contain both clusters.

Required clustering quality:

| Dataset | ARI | NMI |
|---|---:|---:|
| Diabetes | >= 0.34 | >= 0.28 |
| Heart | >= 0.35 | >= 0.28 |

Cluster IDs are arbitrary; ARI and NMI are permutation-invariant.

## Datasets

### Diabetes Dataset

The diabetes dataset contains one numerical feature:

- `age`

and fifteen categorical features:

- `gender`
- `polyuria`
- `polydipsia`
- `sudden_weight_loss`
- `weakness`
- `polyphagia`
- `genital_thrush`
- `visual_blurring`
- `itching`
- `irritability`
- `delayed_healing`
- `partial_paresis`
- `muscle_stiffness`
- `alopecia`
- `obesity`

### Heart Dataset

The heart dataset contains four numerical features:

- `trestbps`
- `chol`
- `thalach`
- `oldpeak`

and eight categorical features:

- `sex`
- `cp`
- `fbs`
- `restecg`
- `exang`
- `slope`
- `ca`
- `thal`

## Solution Overview

The reference solution in `solution/solve.sh` uses the following pipeline:

```text
Raw tabular data
      |
      v
Categorical features --> One-Hot Encoding
Numerical features   --> Standard Scaling
      |
      v
Combined feature matrix
      |
      v
Variational Autoencoder (VAE)
      |
      v
Latent representation
      |
      +--> Spectral Clustering (Diabetes)
      |
      +--> K-Means (Heart)
      |
      v
pseudo_label
```

### Preprocessing

Categorical variables are transformed using the custom one-hot encoder in:

```text
environment/src/transform/OnehotEncoder.py
```

Numerical variables are standardized using:

```text
environment/src/transform/StandardScaler.py
```

### Latent Representation

A PyTorch **Variational Autoencoder (VAE)** is trained independently on each dataset.

The encoder architecture is:

```text
Input
  -> Linear(256)
  -> CELU
  -> Linear(128)
  -> CELU
  -> Linear(64)
  -> CELU
  -> latent mean / latent log-variance
```

The decoder reconstructs the original preprocessed feature vector from the latent representation.

Training uses:

- Adam optimizer
- learning rate: `1e-3`
- batch size: `64`
- epochs: `600`
- random seed: `42`
- CPU execution
- reconstruction loss + KL divergence

The final latent representation uses the encoder mean (`mu`) rather than a sampled latent vector.

### Clustering

The current solution uses:

| Dataset | Latent Dimension | Algorithm |
|---|---:|---|
| Diabetes | 8 | Spectral Clustering |
| Heart | 16 | K-Means |

Implementations are located under:

```text
environment/src/cluster/
```

The repository also includes implementations/wrappers for several other clustering algorithms:

- K-Means
- MiniBatch K-Means
- Bisecting K-Means
- Gaussian Mixture
- Agglomerative Clustering
- Spectral Clustering
- BIRCH

## Project Structure

```text
clustering/
├── instruction.md
├── task.toml
├── environment/
│   ├── diabetes_data.csv
│   ├── heart.csv
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── cluster/
│       │   ├── agglomerative.py
│       │   ├── birch.py
│       │   ├── bisecting_kmeans.py
│       │   ├── gaussian_mixture.py
│       │   ├── kmean.py
│       │   ├── minibatch_kmeans.py
│       │   └── spectral.py
│       └── transform/
│           ├── BinaryEncoder.py
│           ├── CountEncoder.py
│           ├── FrequencyEncoder.py
│           ├── HashEncoder.py
│           ├── KBinsDiscretizer.py
│           ├── MaxAbsScaler.py
│           ├── Normalizer.py
│           ├── OnehotEncoder.py
│           ├── OrdinalEncoder.py
│           ├── PolynomialFeatures.py
│           ├── RobustScaler.py
│           └── StandardScaler.py
├── solution/
│   └── solve.sh
└── tests/
    ├── test.sh
    └── test_outputs.py
```

## Environment

The task runs in a Python 3.13 container with no GPU and no internet access.

Resource limits defined in `task.toml`:

- CPUs: `2`
- Memory: `4096 MB`
- Storage: `10240 MB`
- GPUs: `0`
- Agent timeout: `1700 s`
- Verifier timeout: `150 s`

Main Python dependencies:

```text
numpy==2.2.6
pandas==3.0.3
scikit-learn==1.6.1
torch==2.9.1
pytest==9.1.1
pytest-json-ctrf==0.5.2
pyyaml==6.0.2
```

## Running the Solution

Inside the task container:

```bash
bash /app/solution/solve.sh
```

The script reads:

```text
/app/diabetes_data.csv
/app/heart.csv
```

and writes:

```text
/app/output_diabetes_data.csv
/app/output_heart.csv
```

Example output format:

```csv
pseudo_label
0
1
1
0
...
```

## Running the Tests

Run:

```bash
bash /app/tests/test.sh
```

or execute the Python verifier directly:

```bash
python /app/tests/test_outputs.py
```

The verifier checks:

1. ARI and NMI thresholds.
2. Prediction row counts.
3. Presence of both cluster labels.
4. Exactly one output column.
5. Correct output filenames.
6. Correct `pseudo_label` column name.

## Evaluation Metrics

### Adjusted Rand Index (ARI)

ARI measures agreement between the predicted clustering and the hidden ground-truth partition while correcting for agreement expected by chance.

A perfect clustering has:

```text
ARI = 1.0
```

### Normalized Mutual Information (NMI)

NMI measures the amount of shared information between predicted clusters and ground-truth classes, normalized to the range `[0, 1]`.

A perfect clustering has:

```text
NMI = 1.0
```

Because both metrics are invariant to cluster-label permutations, predicting cluster IDs in the opposite order does not affect the score.

## Extending the Solution

The provided VAE-based approach is only the reference implementation. Alternative preprocessing, representation-learning, or clustering techniques may be used as long as the generated outputs satisfy the required format and evaluation thresholds.

Custom preprocessing utilities can be added under:

```text
environment/src/transform/
```

and clustering implementations can be added under:

```text
environment/src/cluster/
```

## Reproducibility

The reference solution uses a fixed random seed:

```python
SEED = 42
```

NumPy, Python, and PyTorch random generators are seeded to improve reproducibility across runs.

## License

No license is specified in the current repository.
