import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram

# ── Audio features used for clustering ────────────────────────────────────────
FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence',
    'tempo', 'duration_ms',
]


def data():
    """Load and clean the Spotify songs CSV."""
    df = pd.read_csv("spotify_songs.csv", index_col=1)
    df.drop(
        ["track_id", "track_album_id", "playlist_id", "playlist_name", "playlist_subgenre"],
        inplace=True, axis=1,
    )
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    # Normalise year-only release dates like '2012' → '2012-01-01'
    df['track_album_release_date'] = df['track_album_release_date'].astype(str).str.strip()
    mask = df['track_album_release_date'].str.match(r'^\d{4}$')
    df.loc[mask, 'track_album_release_date'] = df.loc[mask, 'track_album_release_date'] + '-01-01'
    df['track_album_release_date'] = pd.to_datetime(
        df['track_album_release_date'], format='mixed'
    )
    return df


def scale_data(df):
    """Scale the audio FEATURES columns with StandardScaler."""
    scaler = StandardScaler()
    X = df[FEATURES].dropna()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, scaler


def run_kmeans(X_scaled, k):
    """Run K-Means and return predicted integer labels."""
    km = KMeans(n_clusters=k, random_state=7, n_init=10)
    return km.fit_predict(X_scaled)


def run_hierarchical(X_scaled):
    """Run Agglomerative Clustering and return the fitted model."""
    clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    clustering.fit(X_scaled)
    return clustering


def run_pca(X_scaled, n_components=2):
    """PCA dimensionality reduction."""
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X_scaled)


def get_dendrogram_fig(model, **kwargs):
    """Build the Matplotlib dendrogram figure from an AgglomerativeClustering model."""
    fig, ax = plt.subplots(figsize=(14, 5))
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1          # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    dendrogram(linkage_matrix, ax=ax, **kwargs)
    ax.set_xlabel("Number of points in node (or index of point if no parenthesis)")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    return fig


def compute_elbow_silhouette(X_scaled, k_range=range(2, 9)):
    """
    Compute KMeans inertia (elbow) and silhouette scores for each k in k_range.

    Returns
    -------
    k_vals        : list[int]
    inertias      : list[float]   (WCSS / within-cluster sum of squares)
    sil_scores    : list[float]
    """
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=7, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))
    return list(k_range), inertias, sil_scores


def assign_clusters(df, X_scaled, k):
    """
    Run K-Means with *k* clusters and return a copy of *df* that has an
    integer 'Cluster' column aligned positionally with X_scaled.
    """
    km = KMeans(n_clusters=k, random_state=7, n_init=10)
    labels = km.fit_predict(X_scaled)
    df_copy = df.copy().reset_index(drop=True)
    df_copy['Cluster'] = labels
    return df_copy


def recommend_playlists(df_with_clusters, selected_songs, k):
    """
    Score every cluster by what fraction of *selected_songs* it contains.

    Parameters
    ----------
    df_with_clusters : DataFrame
        Must have a 'track_name' column and an integer 'Cluster' column.
    selected_songs   : list[str]
        Track names chosen by the user (up to 5).
    k                : int
        Number of clusters (used to initialise the score dict).

    Returns
    -------
    list[dict] sorted by score descending, each entry::

        {
          'cluster'      : int,
          'score'        : float,   # fraction of selected songs in cluster
          'count'        : int,     # number of selected songs in cluster
          'total_tracks' : int,     # total songs in cluster
          'matched_songs': list[str], # list of matched song names
        }
    """
    name_col = 'track_name' if 'track_name' in df_with_clusters.columns else None
    cluster_counts = {i: 0 for i in range(k)}
    cluster_matches = {i: [] for i in range(k)}
    found = 0

    for song in selected_songs:
        if name_col:
            matches = df_with_clusters[df_with_clusters[name_col] == song]
        else:
            matches = df_with_clusters[df_with_clusters.index == song]

        if not matches.empty:
            cid = int(matches.iloc[0]['Cluster'])
            cluster_counts[cid] += 1
            cluster_matches[cid].append(song)
            found += 1

    if found == 0:
        return []

    results = []
    for cid, count in cluster_counts.items():
        tracks_in_cluster = df_with_clusters[df_with_clusters['Cluster'] == cid]
        results.append({
            'cluster':      cid,
            'score':        count / found,
            'count':        count,
            'total_tracks': len(tracks_in_cluster),
            'matched_songs': cluster_matches[cid],
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results


def interpret_cluster(df_with_clusters, X_scaled, cid):
    """
    Provide interpretable insights for a specific cluster.
    
    Returns
    -------
    dict with:
        'distinctive_feature': feature that separates this cluster from others the most
        'distinctive_direction': "higher" or "lower" than average
        'cohesive_feature': feature that is most consistent (lowest variance) in this cluster
    """
    cluster_mask = df_with_clusters['Cluster'] == cid
    X_cluster = X_scaled[cluster_mask]
    
    # 1. Distincting feature
    means = np.mean(X_cluster, axis=0)
    dist_idx = np.argmax(np.abs(means))
    
    # 2. Cohesive feature
    stds = np.std(X_cluster, axis=0)
    coh_idx = np.argmin(stds)
    
    return {
        'distinctive_feature': FEATURES[dist_idx],
        'distinctive_direction': "higher" if means[dist_idx] > 0 else "lower",
        'cohesive_feature': FEATURES[coh_idx]
    }


def compare_outliers(df_with_clusters, X_scaled, selected_songs):
    """
    If selected songs land in multiple clusters, find the feature that best
    separates those specific clusters (highest variance among their centroids).
    """
    clusters_present = []
    name_col = 'track_name' if 'track_name' in df_with_clusters.columns else None
    
    for song in selected_songs:
        if name_col:
            matches = df_with_clusters[df_with_clusters[name_col] == song]
        else:
            matches = df_with_clusters[df_with_clusters.index == song]
            
        if not matches.empty:
            cid = int(matches.iloc[0]['Cluster'])
            if cid not in clusters_present:
                clusters_present.append(cid)
    
    if len(clusters_present) <= 1:
        return None
        
    centroids = []
    for c in clusters_present:
        c_mask = df_with_clusters['Cluster'] == c
        c_mean = np.mean(X_scaled[c_mask], axis=0)
        centroids.append(c_mean)
        
    centroids = np.array(centroids)
    feature_variances = np.var(centroids, axis=0)
    best_feat_idx = np.argmax(feature_variances)
    
    return {
        'clusters': clusters_present,
        'feature': FEATURES[best_feat_idx]
    }
