# Task 5: Movie Recommendation System---------Elevvo Tech Internship-----------
# ---------------------------------------------------------------------
# Build and evaluate a movie recommendation system based on user similarity.
# Includes interactive function for client-side testing.

# ------------------------- 1. Imports -------------------------
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
import joblib
import matplotlib.pyplot as plt
from pandas.plotting import table
from datetime import datetime

# ------------------------- 2. Load & Clean Data ---------------
# Define the base path
base_path = r"C:\Users\arshi\OneDrive\Desktop\Internship_Material\Movie_Recom_Dataset"

# Load ratings and movies data (adjust file names if different)
ratings_path = f"{base_path}\\u.data"
movies_path = f"{base_path}\\u.item"

ratings = pd.read_csv(ratings_path, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
movies = pd.read_csv(movies_path, sep='|', encoding='latin-1', usecols=[0, 1], names=['movie_id', 'title'])

print("Loaded ratings shape:", ratings.shape)
print("Loaded movies shape:", movies.shape, "\n")
# No major cleaning needed; drop timestamp
ratings.drop(columns=['timestamp'], inplace=True)

# Train/test split on ratings
train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)
print("Train ratings shape:", train_ratings.shape)
print("Test ratings shape:", test_ratings.shape, "\n")

# Build user-item matrix for train (sparse, filled with 0)
user_item_train = train_ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
print("User-item matrix shape:", user_item_train.shape, "\n")

# ------------------------- 3. User-Based Similarity -------------------
user_sim = cosine_similarity(user_item_train)
user_sim_df = pd.DataFrame(user_sim, index=user_item_train.index, columns=user_item_train.index)

# ------------------------- 4.  Item-Based Similarity -------------------
item_item_train = user_item_train.T
item_sim = cosine_similarity(item_item_train)
item_sim_df = pd.DataFrame(item_sim, index=item_item_train.index, columns=item_item_train.index)

# ------------------------- 5. Matrix Factorization (SVD) ---------------
svd = TruncatedSVD(n_components=20, random_state=42)
user_factors = svd.fit_transform(user_item_train)
predicted_ratings = pd.DataFrame(np.dot(user_factors, svd.components_), index=user_item_train.index, columns=user_item_train.columns)

# ------------------------- 6. Recommendation Functions ----------------
def get_user_based_recs(user_id, n_sim_users=20, n_recs=5):
    if user_id not in user_item_train.index:
        return pd.Series()  # Empty if user not found
    sim_scores = user_sim_df[user_id].sort_values(ascending=False)[1:n_sim_users + 1]
    similar_users = sim_scores.index
    weighted_ratings = np.dot(sim_scores, user_item_train.loc[similar_users]) / sim_scores.sum()
    pred_ratings = pd.Series(weighted_ratings, index=user_item_train.columns)
    unseen = pred_ratings[user_item_train.loc[user_id] == 0]
    top_recs = unseen.sort_values(ascending=False).head(n_recs)
    return top_recs

def get_item_based_recs(user_id, n_sim_items=20, n_recs=5):
    if user_id not in user_item_train.index:
        return pd.Series()
    user_ratings = user_item_train.loc[user_id]
    pred = {}
    for movie in user_item_train.columns:
        if user_ratings[movie] == 0:
            sim_scores = item_sim_df[movie].sort_values(ascending=False)[1:n_sim_items + 1]
            rel_ratings = user_ratings[sim_scores.index]
            sim_sum = np.sum(np.abs(sim_scores[rel_ratings > 0]))
            if sim_sum > 0:
                pred[movie] = np.dot(sim_scores[rel_ratings > 0], rel_ratings[rel_ratings > 0]) / sim_sum
    if not pred:
        return pd.Series()
    top_recs = pd.Series(pred).sort_values(ascending=False).head(n_recs)
    return top_recs

def get_svd_recs(user_id, n_recs=5):
    if user_id not in predicted_ratings.index:
        return pd.Series()
    unseen = predicted_ratings.loc[user_id][user_item_train.loc[user_id] == 0]
    top_recs = unseen.sort_values(ascending=False).head(n_recs)
    return top_recs

# ------------------------- 7. Evaluation (Precision at K) ---------------------
def precision_at_k(method='user', k=5, threshold=4.0):
    user_precs = []
    test_users = test_ratings['user_id'].unique()
    for i, user in enumerate(test_users):
        if user not in user_item_train.index:
            continue
        if i % 50 == 0:  # Print every 50 users to avoid flooding the console
            print(f"Evaluating {method} for user {user} ({i+1}/{len(test_users)})")
        if method == 'user':
            recs_series = get_user_based_recs(user, n_recs=k)
        elif method == 'item':
            recs_series = get_item_based_recs(user, n_recs=k)
        elif method == 'svd':
            recs_series = get_svd_recs(user, n_recs=k)
        else:
            raise ValueError("Invalid method")
        recs = recs_series.index.tolist()
        user_test = test_ratings[(test_ratings['user_id'] == user) & (test_ratings['rating'] >= threshold)]
        relevant = set(user_test['movie_id'])
        hits = len(set(recs) & relevant)
        prec = hits / k if recs else 0
        user_precs.append(prec)
    return np.mean(user_precs)

print("Precision@5 (User-Based):", round(precision_at_k('user', 5), 3))
print("Precision@5 (Item-Based):", round(precision_at_k('item', 5), 3))
print("Precision@5 (SVD):", round(precision_at_k('svd', 5), 3), "\n")

# ------------------------- 8. Save Models ---------------------
joblib.dump({
    'user_item_train': user_item_train,
    'user_sim_df': user_sim_df,
    'item_sim_df': item_sim_df,
    'predicted_ratings': predicted_ratings,
    'movies': movies,
    'svd': svd
}, f"{base_path}\\movie_rec_models.joblib")
print("Models saved: movie_rec_models.joblib\n")

# ------------------------- 9. Interactive Client Testing ------
def predict_recs(user_id: int, method: str = 'user', n_recs: int = 5) -> pd.DataFrame:
    """
    Generate movie recommendations for a single user.
    method: 'user', 'item', or 'svd'
    """
    saved = joblib.load(f"{base_path}\\movie_rec_models.joblib")
    user_item_train = saved['user_item_train']
    movies = saved['movies']
    if method == 'user':
        recs_series = get_user_based_recs(user_id, n_recs=n_recs)
    elif method == 'item':
        recs_series = get_item_based_recs(user_id, n_recs=n_recs)
    elif method == 'svd':
        recs_series = get_svd_recs(user_id, n_recs=n_recs)
    else:
        raise ValueError("Invalid method")
    if recs_series.empty:
        return pd.DataFrame()  # No recs
    recs_df = pd.DataFrame({'movie_id': recs_series.index, 'predicted_rating': recs_series.values})
    recs_df = recs_df.merge(movies, on='movie_id')
    return recs_df[['title', 'predicted_rating']]

# ------------------------- 10. Interactive Console Prediction -------------------------
# Load saved models
saved = joblib.load(f"{base_path}\\movie_rec_models.joblib")

print("\n--- Movie Recommendation Entry ---")
user_id = int(input("Enter User ID (1-943): "))
method = input("Choose method (user / item / svd): ").strip().lower()

# Get recommendations
recs_df = predict_recs(user_id, method=method, n_recs=5)

if recs_df.empty:
    print("No recommendations available for this user.")
else:
    # Prepare result table
    recs_df['predicted_rating'] = recs_df['predicted_rating'].round(2)
    recs_df.insert(0, 'Rank', range(1, len(recs_df) + 1))
    result_table = recs_df.rename(columns={'title': 'Movie Title', 'predicted_rating': 'Predicted Rating'})

    print("\n--- Recommendation Results ---")
    # ------------------------- 11. Visual Table -------------------------
    # Build a visual table using matplotlib
    fig, ax = plt.subplots(figsize=(10, len(result_table) + 2))
    ax.axis("off")

    tbl = table(ax, result_table, loc="center", cellLoc="center", colWidths=[0.1, 0.6, 0.3])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 1.5)

    # Highlight rows based on rating
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:  # Header
            cell.set_facecolor("#1976d2")
            cell.set_text_props(color="white", weight="bold")
        elif c == 2:  # Predicted Rating column
            rating = result_table.iloc[r - 1]['Predicted Rating']
            if rating >= 4.5:
                cell.set_facecolor("#4caf50")  # Green for high
            elif rating >= 4.0:
                cell.set_facecolor("#81c784")  # Light green
            else:
                cell.set_facecolor("#ffeb3b")  # Yellow for medium
            cell.set_text_props(weight="bold", color="black")

    ax.set_title(f"Top Movie Recommendations for User {user_id} ({method.capitalize()}-Based)", fontsize=14, weight="bold", pad=20)
    plt.tight_layout()
    plt.show()