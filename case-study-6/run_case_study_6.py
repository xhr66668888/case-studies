"""
Case Study 6 — Dimensionality Reduction on the UCI Wine Dataset
Generates all figures and prints all data needed for the report.
"""
import os
os.environ["NUMBA_OPT"] = "0"  # work around LLVM JIT crash on ARM64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Load Wine dataset
# ============================================================
data = load_wine(as_frame=True)
df = data.frame.copy()

print("=" * 60)
print("Dataset info")
print("=" * 60)
print("Shape:", df.shape)
print("Feature columns:", data.feature_names)
print("Target names:", data.target_names)
print(df.head())

# ============================================================
# Train/test split (70/30, stratified)
# ============================================================
X = df.drop(columns=["target"]).values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain shape: {X_train.shape}  Test shape: {X_test.shape}")
print("Train label counts:", np.bincount(y_train))
print("Test  label counts:", np.bincount(y_test))

# ============================================================
# Helper: evaluate model
# ============================================================
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    yhat_tr = model.predict(X_tr)
    yhat_te = model.predict(X_te)
    tr_acc = accuracy_score(y_tr, yhat_tr)
    te_acc = accuracy_score(y_te, yhat_te)

    print("\n" + "=" * 78)
    print(name)
    print("-" * 78)
    print(f"Train accuracy: {tr_acc:.4f}")
    print(f"Test  accuracy: {te_acc:.4f}")
    print("\n[TEST] Classification report:")
    print(classification_report(y_te, yhat_te, digits=4, zero_division=0))
    print("[TEST] Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_te, yhat_te))
    return {"name": name, "train_acc": tr_acc, "test_acc": te_acc}

# ============================================================
# Helper: plot decision regions (light version)
# ============================================================
def plot_decision_regions_2d_light(ax, X2, y, clf, title, grid_n=180):
    X2 = X2.astype(np.float32, copy=False)
    y  = y.astype(np.int32, copy=False)
    clf.fit(X2, y)

    pad = 0.7
    x_min, x_max = X2[:, 0].min() - pad, X2[:, 0].max() + pad
    y_min, y_max = X2[:, 1].min() - pad, X2[:, 1].max() + pad

    xx = np.linspace(x_min, x_max, grid_n, dtype=np.float32)
    yy = np.linspace(y_min, y_max, grid_n, dtype=np.float32)
    XX, YY = np.meshgrid(xx, yy)
    grid = np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float32, copy=False)
    Z = clf.predict(grid).reshape(XX.shape)

    ax.contourf(XX, YY, Z, alpha=0.18)
    markers = ["^", "s", "o"]
    for c in np.unique(y):
        ax.scatter(X2[y == c, 0], X2[y == c, 1],
                   marker=markers[int(c) % len(markers)], s=28,
                   label=f"class {int(c)+1}")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best")

# ============================================================
# Part I: 3 Logistic Regression pipelines  (Q3)
# ============================================================
print("\n" + "#" * 78)
print("# Part I: Logistic Regression pipelines")
print("#" * 78)

lr = LogisticRegression(max_iter=5000)

pipe_lr_unscaled = Pipeline(steps=[("clf", lr)])
pipe_pca_unscaled = Pipeline(steps=[
    ("pca", PCA(n_components=2, random_state=RANDOM_STATE)),
    ("clf", lr),
])
pipe_scaled_pca = Pipeline(steps=[
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=2, random_state=RANDOM_STATE)),
    ("clf", lr),
])

results = []
results.append(evaluate_model("1) LR (unscaled, 13D)", pipe_lr_unscaled,
                               X_train, y_train, X_test, y_test))
results.append(evaluate_model("2) PCA(2) unscaled -> LR", pipe_pca_unscaled,
                               X_train, y_train, X_test, y_test))
results.append(evaluate_model("3) StandardScaler -> PCA(2) -> LR", pipe_scaled_pca,
                               X_train, y_train, X_test, y_test))

summary = pd.DataFrame(results)[["name", "train_acc", "test_acc"]]
print("\n--- Summary ---")
print(summary.to_string(index=False))

# ============================================================
# PCA visualization under 4 scalers  (Q3 figure)
# ============================================================
print("\nGenerating PCA visualization under 4 scalers...")

scalers = {
    "No scaling": None,
    "Min-Max scaled": MinMaxScaler(),
    "Standard scaled": StandardScaler(),
    "Robust scaled": RobustScaler(),
}
lr_vis = LogisticRegression(max_iter=3000)

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, (label, scaler) in zip(axes, scalers.items()):
    if scaler is None:
        X_proc = X.astype(np.float32, copy=False)
    else:
        X_proc = scaler.fit_transform(X).astype(np.float32, copy=False)
    X_pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_proc)
    X_pca = X_pca.astype(np.float32, copy=False)
    plot_decision_regions_2d_light(ax, X_pca, y, lr_vis,
                                   f"PCA (2D): {label}", grid_n=160)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "q3_pca_scaling_comparison.png"), dpi=150)
plt.close(fig)
print(f"  -> saved figures/q3_pca_scaling_comparison.png")

# ============================================================
# Part II: Kernel PCA (RBF) — effect of gamma  (Q4)
# ============================================================
print("\n" + "#" * 78)
print("# Part II-a: Kernel PCA (RBF) — gamma comparison")
print("#" * 78)

scaler_std = StandardScaler()
Xtr_s = scaler_std.fit_transform(X_train)
Xte_s = scaler_std.transform(X_test)

gammas = [0.1, 0.5, 1]

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
fig.suptitle("Kernel PCA (RBF) — effect of gamma (test set)", fontsize=13)

for ax, g in zip(axes, gammas):
    kpca = KernelPCA(n_components=2, kernel="rbf", gamma=g,
                     fit_inverse_transform=False)
    kpca.fit(Xtr_s)
    Z_te = kpca.transform(Xte_s)

    markers = ["^", "s", "o"]
    for c in np.unique(y_test):
        ax.scatter(Z_te[y_test == c, 0], Z_te[y_test == c, 1],
                   marker=markers[int(c)], s=40, label=f"class {int(c)+1}")
    ax.set_title(f"gamma = {g}")
    ax.set_xlabel("1st component")
    ax.set_ylabel("2nd component")
    ax.legend(loc="best")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "q4_kernel_pca_gamma.png"), dpi=150)
plt.close(fig)
print(f"  -> saved figures/q4_kernel_pca_gamma.png")

# ============================================================
# Part II-b: LDA  (bonus, for completeness)
# ============================================================
print("\n" + "#" * 78)
print("# Part II-b: LDA")
print("#" * 78)

lda = LDA(n_components=2)
Z_lda = lda.fit_transform(Xtr_s, y_train)
Z_lda_te = lda.transform(Xte_s)

fig, ax = plt.subplots(figsize=(6, 4))
markers = ["^", "s", "o"]
for c in np.unique(y_test):
    ax.scatter(Z_lda_te[y_test == c, 0], Z_lda_te[y_test == c, 1],
               marker=markers[int(c)], s=35, label=f"class {int(c)+1}")
ax.set_title("LDA on standardized data (test set)")
ax.set_xlabel("1st component")
ax.set_ylabel("2nd component")
ax.legend(loc="best")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "q3_lda.png"), dpi=150)
plt.close(fig)
print(f"  -> saved figures/q3_lda.png")

# ============================================================
# Part II-c: UMAP — parameter grid  (Q5)
# ============================================================
print("\n" + "#" * 78)
print("# Part II-c: UMAP — n_neighbors vs min_dist")
print("#" * 78)

import umap

n_neighbors_list = [5, 10, 15]
min_dist_list    = [0.1, 0.5, 1.0]

fig, axes = plt.subplots(3, 3, figsize=(15, 13))
fig.suptitle("UMAP — n_neighbors vs min_dist (test set)", fontsize=13)

markers = ["^", "s", "o"]

for row, nn in enumerate(n_neighbors_list):
    for col, md in enumerate(min_dist_list):
        ax = axes[row][col]
        um = umap.UMAP(n_components=2, random_state=RANDOM_STATE,
                       n_neighbors=nn, min_dist=md)
        um.fit(Xtr_s, y_train)
        Z_te = um.transform(Xte_s)

        for c in np.unique(y_test):
            ax.scatter(Z_te[y_test == c, 0], Z_te[y_test == c, 1],
                       marker=markers[int(c)], s=35, label=f"class {int(c)+1}")
        ax.set_title(f"n_neighbors={nn}, min_dist={md}")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        if row == 0 and col == 0:
            ax.legend(loc="best")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "q5_umap_grid.png"), dpi=150)
plt.close(fig)
print(f"  -> saved figures/q5_umap_grid.png")

print("\n" + "=" * 60)
print("All done! Figures saved to:", OUT_DIR)
print("=" * 60)
