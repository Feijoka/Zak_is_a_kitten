import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from backend import data, scale_data, run_kmeans, run_hierarchical, run_pca, get_dendrogram_fig
from sklearn.preprocessing import StandardScaler

# Streamlit App Configuration
st.set_page_config(page_title="Spotify Songs Analysis", layout="wide", page_icon="🎵")

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    h1, h2, h3 {color: #1DB954;}
    .stSlider > div > div > div {background: #1DB954 !important;}
    </style>
""", unsafe_allow_html=True)

st.title("🎵 Spotify Songs - Unsupervised Learning")
st.markdown("Explore patterns in Spotify songs using **K-Means** and **Hierarchical Clustering**.")

@st.cache_data
def load_and_scale():
    df = data().reset_index()
    X, X_scaled, scaler = scale_data(df)
    df = df.loc[X.index]
    return df, X, X_scaled, scaler

# Load Data
df, X, X_scaled, scaler = load_and_scale()

# Sidebar
st.sidebar.header("Dataset Options")
if st.sidebar.checkbox("Show Raw Dataset", False):
    st.subheader("Raw Data Sample")
    st.dataframe(df.head(50))
st.sidebar.markdown("---")

# Tabs for visual separation
tab1, tab2 = st.tabs(["K-Means & PCA", "Hierarchical Clustering"])

with tab1:
    st.header("K-Means Clustering Analysis")
    
    k = st.sidebar.slider("Number of Clusters (k) for K-Means", min_value=2, max_value=8, value=2, step=1)
    
    # Backend call to KMeans
    labels = run_kmeans(X_scaled, k)
    df['Cluster'] = labels.astype(str)
    
    # Backend call to PCA
    components = run_pca(X_scaled, 2)
    df['PCA1'] = components[:, 0]
    df['PCA2'] = components[:, 1]
    
    sample_size = st.sidebar.slider("Sample size for PCA Plot", 500, 10000, 2000, 500)
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    fig_scatter = px.scatter(
        df_sample, 
        x="PCA1", 
        y="PCA2", 
        color="Cluster",
        hover_data=['track_name', 'track_artist'],
        title=f"2D PCA Visualization (k={k})",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_scatter.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("### Cluster Profiles Heatmap")
    cluster_means = df.groupby('Cluster')[X.columns].mean()
    scaler_means = StandardScaler()
    cluster_means_scaled = pd.DataFrame(
        scaler_means.fit_transform(cluster_means), 
        columns=cluster_means.columns, 
        index=cluster_means.index
    )
    
    fig_heat = px.imshow(
        cluster_means_scaled.T,
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Relative Feature Importance per Cluster"
    )
    fig_heat.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_heat, use_container_width=True)

with tab2:
    st.header("Hierarchical Clustering")
    st.markdown("Dendrogram visualization generated using **Agglomerative Clustering** with distance thresholds.")
    
    @st.cache_resource
    def cached_hierarchical(_X_scaled_data):
        return run_hierarchical(_X_scaled_data)
        
    with st.spinner("Running Agglomerative Clustering (This may take a minute for 30k rows)..."):
        # We sample it slightly if preferred, but let's run completely on X_scaled as requested in notebook.
        hc_model = cached_hierarchical(X_scaled)
        fig_dendro = get_dendrogram_fig(hc_model, truncate_mode="level", p=10)
        st.pyplot(fig_dendro)
