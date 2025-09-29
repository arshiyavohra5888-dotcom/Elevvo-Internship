# Task: Customer Segmentation – Elevvo Tech Internship
# ---------------------------------------------------------------------
# Clean, Visualisation  (matplotlib)
# - K-Means with automatic k selection (silhouette)
# - DBSCAN bonus
# - Attractive scatter, PCA view, centroid annotations, cluster stats
# - Interactive testing dasboard 

# ------------------------- 1. Imports -------------------------
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib import patches
from pandas.plotting import table
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
import os

warnings.filterwarnings("ignore")

# ------------------------- 2. Config & Data Load -------------------------

DATA_PATH = "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv"

def load_data(path=DATA_PATH):
    try:
        df_local = pd.read_csv(path)
        print(f"Loaded data from: {path}")
        return df_local
    except Exception as e:
        print("Could not load from URL. Trying a local file path fallback if it exists.")
        
        fallback = r"C:\Users\arshi\OneDrive\Desktop\Internship_Material\Mall_Customers.csv"
        if os.path.exists(fallback):
            print(f"Loaded local file: {fallback}")
            return pd.read_csv(fallback)
        raise RuntimeError("Could not load dataset. Update DATA_PATH to a valid CSV location.") from e

df = load_data()
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
print("Dataset shape:", df.shape)
print(df.columns.tolist())

# ------------------------- 3. Feature selection -------------------------
# Find columns containing 'income' and 'spending' (common Mall dataset scheme)
income_col = [c for c in df.columns if "income" in c]
spend_col = [c for c in df.columns if "spending" in c or "score" in c]

if not income_col or not spend_col:
    raise ValueError("Couldn't find columns for income or spending. Check dataset columns.")
income_col = income_col[0]
spend_col = spend_col[0]

X_raw = df[[income_col, spend_col]].copy()
X_raw.columns = ["annual_income", "spending_score"]

# ------------------------- 4. Styling for attractive plots -------------------------
plt.rcParams.update({
    "figure.facecolor": "#fbfbfb",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#dcdcdc",
    "axes.grid": True,
    "grid.color": "#eeeeee",
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "legend.frameon": False
})

cmap = plt.cm.get_cmap("tab10")  # clean qualitative colors

# ------------------------- 5. Scaling -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ------------------------- 6. Exploratory scatter -------------------------
def attractive_scatter(x_raw, x_scaled):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    ax[0].scatter(x_raw["annual_income"], x_raw["spending_score"],
                  s=70, alpha=0.85, edgecolor="#2b2b2b", linewidth=0.6)
    ax[0].set_title("Raw: Annual Income vs Spending Score", fontsize=13, fontweight="bold")
    ax[0].set_xlabel("Annual Income")
    ax[0].set_ylabel("Spending Score")

    ax[1].scatter(x_scaled[:, 0], x_scaled[:, 1],
                  s=70, alpha=0.85, edgecolor="#2b2b2b", linewidth=0.6)
    ax[1].set_title("Scaled features (StandardScaler)", fontsize=13, fontweight="bold")
    ax[1].set_xlabel("Income (scaled)")
    ax[1].set_ylabel("Spending (scaled)")
    plt.show()

attractive_scatter(X_raw, X_scaled)

# ------------------------- 7. Choose optimal k (Silhouette + Elbow visuals) -------------------------
def choose_k(X_scaled, k_min=2, k_max=10):
    ks = list(range(k_min, k_max + 1))
    inertias = []
    silhouettes = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    # Plot side-by-side with polished style
    fig, ax = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    ax[0].plot(ks, inertias, marker="o", linewidth=2)
    ax[0].set_title("Elbow Method — Inertia", fontsize=12, fontweight="bold")
    ax[0].set_xlabel("k")
    ax[0].set_ylabel("Inertia")

    ax[1].plot(ks, silhouettes, marker="o", linewidth=2)
    ax[1].set_title("Silhouette Scores", fontsize=12, fontweight="bold")
    ax[1].set_xlabel("k")
    ax[1].set_ylabel("Silhouette Score")
    plt.show()

    best_k = ks[int(np.argmax(silhouettes))]
    print("Silhouettes:", list(zip(ks, [round(s, 3) for s in silhouettes])))
    print("Chosen best_k (max silhouette):", best_k)
    return best_k

best_k = choose_k(X_scaled, 2, 10)

# ------------------------- 8. Fit final KMeans and save pipeline -------------------------
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
k_labels = kmeans.fit_predict(X_scaled)
X_raw["cluster_kmeans"] = k_labels

pipeline_save = {"scaler": scaler, "kmeans": kmeans, "features": ["annual_income", "spending_score"]}
joblib.dump(pipeline_save, "customer_segmentation_kmeans.joblib")
print("Saved pipeline: customer_segmentation_kmeans.joblib")

# ------------------------- 9. PCA 2D visualization  -------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

def plot_clusters_pca(X_pca, labels, kmeans, title="K-Means clusters (PCA view)"):
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    unique = sorted(np.unique(labels))
    for i, cl in enumerate(unique):
        mask = labels == cl
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   s=90, alpha=0.85, label=f"Cluster {cl}",
                   edgecolor="#222222", linewidth=0.6, zorder=2, cmap=cmap)
    # centroids in PCA space
    centers_pca = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1],
               marker="X", s=240, c="#111111", label="Centroids", zorder=3)
    # annotate centroids with cluster numbers
    for i, c in enumerate(centers_pca):
        ax.text(c[0], c[1], f"  C{i}", fontsize=11, weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#bbbbbb", alpha=0.9))
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2")
    ax.legend(frameon=False)
    plt.show()

plot_clusters_pca(X_pca, k_labels, kmeans, title=f"K-Means Clusters (k={best_k}) — PCA View")

# ------------------------- 10. Cluster summary stats & beautiful bar chart -------------------------
cluster_stats = X_raw.groupby("cluster_kmeans").agg(
    size=("annual_income", "count"),
    avg_income=("annual_income", "mean"),
    avg_spending=("spending_score", "mean")
).reset_index().sort_values("cluster_kmeans")

print("\nCluster summary:\n", cluster_stats)

def plot_avg_spending(cluster_stats):
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bars = ax.bar(cluster_stats["cluster_kmeans"].astype(str), cluster_stats["avg_spending"],
                  edgecolor="#222222", linewidth=0.8)
    # highlight cluster with highest avg spending
    top_idx = int(cluster_stats["avg_spending"].idxmax())
    for i, b in enumerate(bars):
        if i == top_idx:
            b.set_alpha(1.0)
            b.set_linestyle("solid")
            b.set_edgecolor("#000000")
            b.set_linewidth(1.6)
        else:
            b.set_alpha(0.8)
    ax.set_title("Average Spending Score per Cluster", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cluster"); ax.set_ylabel("Avg Spending Score")
    # annotate bar values
    for rect in bars:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=10)
    plt.show()

plot_avg_spending(cluster_stats)

# ------------------------- 11.  DBSCAN quick try -------------------------
db = DBSCAN(eps=0.6, min_samples=5)
db_labels = db.fit_predict(X_scaled)
n_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
print(f"\nDBSCAN found {n_db} clusters (noise labeled as -1).")

def plot_dbscan_pca(X_pca, db_labels):
    fig, ax = plt.subplots(figsize=(8.5, 6), constrained_layout=True)
    unique = np.unique(db_labels)
    palette2 = plt.cm.get_cmap("tab20", len(unique))
    for i, lbl in enumerate(unique):
        mask = db_labels == lbl
        name = "Noise" if lbl == -1 else f"Cluster {lbl}"
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=60, alpha=0.8, label=name, edgecolor="#222222")
    ax.set_title("DBSCAN (PCA view)", fontsize=13, fontweight="bold")
    ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2")
    ax.legend(frameon=False)
    plt.show()

plot_dbscan_pca(X_pca, db_labels)

# ------------------------- 12. Interactive Customer Test -------------------------
def interactive_customer_test(model_path="customer_segmentation_kmeans.joblib"):
    """
    Enter income & spending score in console.
    Shows:
     - Pretty result table
     - Cluster distribution donut
     - Avg spending bar with highlighted cluster
    """
    saved = joblib.load(model_path)
    scaler = saved["scaler"]
    kmeans = saved["kmeans"]

    print("\n--- Customer Entry (Interactive Test) ---")
    try:
        income = float(input("Annual Income (enter 60 for 60k if dataset in 'k'): ").strip())
        spending = float(input("Spending Score (1-100): ").strip())
    except Exception as e:
        print("Invalid inputs. Try again with numeric values.")
        return

    sample = np.array([[income, spending]])
    sample_scaled = scaler.transform(sample)
    pred_cluster = int(kmeans.predict(sample_scaled)[0])

    stats = cluster_stats[cluster_stats["cluster_kmeans"] == pred_cluster].squeeze()
    assigned_size = int(stats["size"])
    assigned_avg_income = float(stats["avg_income"])
    assigned_avg_spending = float(stats["avg_spending"])

    # Build result table DataFrame
    result_table = pd.DataFrame({
        "Feature": [
            "Annual Income",
            "Spending Score",
            "Assigned Cluster",
            "Cluster Size",
            "Cluster Avg Income",
            "Cluster Avg Spending"
        ],
        "Value": [
            f"{income}",
            f"{spending}",
            f"Cluster {pred_cluster}",
            f"{assigned_size}",
            f"{assigned_avg_income:.2f}",
            f"{assigned_avg_spending:.2f}"
        ]
    })

    # Build dashboard figure
    fig = plt.figure(figsize=(11, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # Table panel
    ax_table = fig.add_subplot(gs[:, 0])
    ax_table.axis("off")
    tbl = table(ax_table, result_table, loc="center", cellLoc="left", colWidths=[0.6, 0.4])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.15, 1.15)

    # Style assigned row
    for (r, c), cell in tbl.get_celld().items():
        if r > 0:
            feature = result_table.iloc[r - 1]["Feature"]
            if feature == "Assigned Cluster":
                cell.set_facecolor("#fff3bf")  # soft yellow
                cell.set_text_props(weight="bold")
            if feature == "Cluster Avg Spending":
                cell.set_facecolor("#dcedc8")  # soft green
                cell.set_text_props(weight="bold")

    ax_table.set_title("Customer Segment Prediction", fontsize=14, fontweight="bold", pad=14)

    # Donut chart: cluster distribution
    ax_donut = fig.add_subplot(gs[0, 1:])
    counts = cluster_stats.set_index("cluster_kmeans")["size"]
    wedges, texts, autotexts = ax_donut.pie(counts.values, labels=[f"C{int(i)} ({int(s)})" for i, s in zip(counts.index, counts.values)],
                                           autopct="%1.0f%%", startangle=140, pctdistance=0.75)
    # draw center circle for donut style
    centre_circle = plt.Circle((0, 0), 0.55, fc="#fbfbfb")
    ax_donut.add_artist(centre_circle)
    ax_donut.set_title("Cluster Distribution (population)", fontsize=12)

    # Bar chart: avg spending highlighting assigned cluster
    ax_bar = fig.add_subplot(gs[1, 1:])
    bars = ax_bar.bar(cluster_stats["cluster_kmeans"].astype(str), cluster_stats["avg_spending"], edgecolor="#222222", linewidth=0.7)
    for b, cl in zip(bars, cluster_stats["cluster_kmeans"]):
        if int(cl) == pred_cluster:
            b.set_linewidth(2.0)
            b.set_edgecolor("#000000")
            b.set_alpha(1.0)
    ax_bar.set_xlabel("Cluster"); ax_bar.set_ylabel("Avg Spending")
    ax_bar.set_title("Average Spending by Cluster")
    for rect in bars:
        ax_bar.annotate(f"{rect.get_height():.1f}", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    # Overall title
    fig.suptitle(f"Customer assigned to Cluster {pred_cluster} — Interactive Summary", fontsize=15, fontweight="bold")
    plt.show()

    # Console summary
    print("\n--- Summary ---")
    print(f"Input -> Income: {income} | Spending: {spending}")
    print(f"Assigned Cluster: {pred_cluster}")
    print(f"Cluster size: {assigned_size} | Avg spending: {assigned_avg_spending:.2f}")

# ------------------------- 13. Save labeled dataset for exploration -------------------------
out_name = "mall_customers_with_clusters.csv"
df_out = df.copy()
df_out["cluster_kmeans"] = X_raw["cluster_kmeans"]
df_out.to_csv(out_name, index=False)
print(f"\nSaved labeled dataset: {out_name}")

# ------------------------- 14. Quick usage instructions -------------------------
print("\nTo run the interactive customer test, in the same Python session call:")
print("    interactive_customer_test()")
print("Or run this script, then import the saved joblib and call interactive_customer_test().")
## ------------------------- 15. Auto-launch Interactive Test -------------------------
if __name__ == "__main__":
    # Ask user if they want to try the interactive test right now
    ans = input("\nWould you like to test a new customer now? (y/n): ").strip().lower()
    if ans == "y":
        interactive_customer_test("customer_segmentation_kmeans.joblib")
    else:
        print("Okay. You can later run in a Python shell:\n    interactive_customer_test()")

# run_customer_test.py
# ---------------------------------------------------------------
# Quick launcher to run the interactive test right away.

from customer_segmentation import interactive_customer_test

# Call the test function directly
interactive_customer_test("customer_segmentation_kmeans.joblib")
# ------------------- End of Script -------------------------