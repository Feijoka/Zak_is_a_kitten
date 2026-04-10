import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram

def data():
    data=pd.read_csv("spotify_songs.csv", index_col=1)
    data.drop(["track_id", "track_album_id", "playlist_id", "playlist_name", "playlist_subgenre"], inplace=True, axis=1)
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)
    # Normalize year-only release dates like '2012' to '2012-01-01'
    data['track_album_release_date'] = data['track_album_release_date'].astype(str).str.strip()
    mask = data['track_album_release_date'].str.match(r'^\d{4}$')
    data.loc[mask, 'track_album_release_date'] = data.loc[mask, 'track_album_release_date'] + '-01-01'
    # Now parse all dates using YYYY-MM-DD format
    data['track_album_release_date']=pd.to_datetime(data['track_album_release_date'], format='mixed')
    return data

FEATURES = ['danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', "duration_ms"]

def data():
    df = pd.read_csv("spotify_songs.csv", index_col=1)
    df.drop(["track_id", "track_album_id", "playlist_id", "playlist_name", "playlist_subgenre"], inplace=True, axis=1)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df['track_album_release_date'] = df['track_album_release_date'].astype(str).str.strip()
    mask = df['track_album_release_date'].str.match(r'^\d{4}$')
    df.loc[mask, 'track_album_release_date'] = df.loc[mask, 'track_album_release_date'] + '-01-01'
    df['track_album_release_date'] = pd.to_datetime(df['track_album_release_date'], format='mixed')
    return df

def scale_data(data):
    """Scales data using StandardScaler for specified features."""
    scaler = StandardScaler()
    X = data[FEATURES].dropna()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, scaler

def run_kmeans(X_scaled, k):
    """Runs K-Means clustering and returns predicted labels."""
    km = KMeans(n_clusters=k, random_state=7, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels

def run_hierarchical(X_scaled):
    """Runs Agglomerative Clustering and returns the model."""
    clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    clustering.fit(X_scaled)
    return clustering

def run_pca(X_scaled, n_components=2):
    """Runs Principal Component Analysis for dimensionality reduction."""
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)
    return components

def get_dendrogram_fig(model, **kwargs):
    """Generates a matplotlib figure of the dendrogram given a hierarchical model."""
    fig, ax = plt.subplots(figsize=(14, 5))
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, ax=ax, **kwargs)
    ax.set_xlabel("Number of points in node (or index of point if no parenthesis)")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    return fig
