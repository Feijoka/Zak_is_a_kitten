import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

import cache_manager as cm
from backend import (
    data as _load_csv, scale_data, run_kmeans, run_hierarchical,
    run_pca, compute_elbow_silhouette,
    get_dendrogram_fig, recommend_playlists, FEATURES,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Spotify Songs Analysis", layout="wide", page_icon="🎵")

st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    h1, h2, h3 { color: #1DB954; }
    .stSlider > div > div > div { background: #1DB954 !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🎵 Spotify Songs — Unsupervised Learning")
st.markdown(
    "Explore patterns in Spotify songs using **K-Means** and **Hierarchical Clustering**."
)


# ── First-time computation (runs once, then saved to disk) ────────────────────

def _compute_and_save():
    """Run every heavy computation and persist results to disk."""
    with st.status("⚙️ First-time setup — computing all models…", expanded=True) as status:

        status.write("📂  Loading and cleaning dataset…")
        _df = _load_csv().reset_index()
        _X, _X_scaled, _scaler = scale_data(_df)
        _df = _df.loc[_X.index].reset_index(drop=True)

        status.write("🎯  Running K-Means (k = 2 → 8)…")
        _labels = {k: run_kmeans(_X_scaled, k) for k in range(2, 9)}

        status.write("📐  Computing PCA components…")
        _pca = run_pca(_X_scaled, 2)

        status.write("📈  Computing Elbow & Silhouette scores…")
        _k_v, _inel, _sils = compute_elbow_silhouette(_X_scaled, range(2, 9))

        status.write("🌲  Running Agglomerative Clustering (slowest step — ~1 min)…")
        _hc = run_hierarchical(_X_scaled)

        status.write("💾  Saving everything to disk…")
        cm.save_all(_df, _X, _X_scaled, _scaler, _labels,
                    _pca, _k_v, _inel, _sils, _hc)

        status.update(
            label="✅  All models computed and saved — future launches are instant!",
            state="complete",
        )


# Trigger first-time build if the disk cache is missing or stale
if not cm.cache_is_valid():
    _compute_and_save()
    st.cache_resource.clear()   # ensure the loader below re-reads from disk


# ── Load everything from disk (singleton — loaded once per process) ────────────
@st.cache_resource(show_spinner="Loading cached models from disk…")
def _load_everything():
    return cm.load_all()


(df, X, X_scaled, scaler,
 labels_dict, pca_components,
 k_vals, inertias, sil_scores,
 hc_model) = _load_everything()


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Dataset Options")

if st.sidebar.checkbox("Show Raw Dataset", False):
    st.subheader("Raw Data Sample")
    st.dataframe(df.head(50))

st.sidebar.markdown("---")

if st.sidebar.button("🗑️ Clear disk cache & recompute"):
    cm.clear_cache()
    st.cache_resource.clear()
    st.rerun()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA",
    "K-Means & PCA",
    "Hierarchical Clustering",
    "🎵 Playlist Recommender",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Exploratory Data Analysis")

    # KPI metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Songs", f"{len(df):,}")
    c2.metric("Audio Features", len(FEATURES))
    c3.metric(
        "Unique Genres",
        df['playlist_genre'].nunique() if 'playlist_genre' in df.columns else "N/A",
    )
    c4.metric("Missing Values (audio)", int(df[FEATURES].isnull().sum().sum()))
    st.markdown("---")

    # Genre distribution
    if 'playlist_genre' in df.columns:
        st.subheader("Genre Distribution")
        genre_counts = df['playlist_genre'].value_counts().reset_index()
        genre_counts.columns = ['Genre', 'Count']
        fig_genre = px.bar(
            genre_counts, x='Genre', y='Count',
            color='Count', color_continuous_scale='Viridis',
            title="Number of Songs per Genre",
        )
        fig_genre.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_genre, use_container_width=True)
    st.markdown("---")

    # Feature distributions
    st.subheader("Audio Feature Distributions")
    selected_feature = st.selectbox("Select a feature to explore", FEATURES)
    col_hist, col_box = st.columns(2)

    with col_hist:
        fig_hist = px.histogram(
            df, x=selected_feature, nbins=60,
            color_discrete_sequence=['#1DB954'],
            title=f"Distribution of {selected_feature}",
        )
        fig_hist.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_box:
        if 'playlist_genre' in df.columns:
            fig_box = px.box(
                df, x='playlist_genre', y=selected_feature,
                color='playlist_genre',
                color_discrete_sequence=px.colors.qualitative.Set2,
                title=f"{selected_feature} by Genre",
            )
            fig_box.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig_box, use_container_width=True)
    st.markdown("---")

    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap")
    corr = df[FEATURES].corr()
    fig_corr = px.imshow(
        corr, color_continuous_scale='RdBu', zmin=-1, zmax=1,
        title="Audio Feature Correlations", text_auto=".2f",
    )
    fig_corr.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown("---")

    # Elbow + Silhouette (pre-computed, loaded from disk)
    st.subheader("Optimal k — Elbow & Silhouette Analysis")
    st.markdown(
        "These plots help determine the optimal number of clusters for K-Means."
    )

    best_k = k_vals[int(np.argmax(sil_scores))]

    col_sil, col_elbow = st.columns(2)
    with col_sil:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=k_vals, y=sil_scores, mode='lines+markers',
            line=dict(color='steelblue', width=2), marker=dict(size=8),
        ))
        fig_sil.add_vline(x=best_k, line_dash='dash', line_color='tomato',
                          annotation_text=f'Best k = {best_k}',
                          annotation_position='top right')
        fig_sil.update_layout(
            title='Silhouette Score vs. k',
            xaxis_title='Number of clusters (k)', yaxis_title='Silhouette Score',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    with col_elbow:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=k_vals, y=inertias, mode='lines+markers',
            line=dict(color='darkorange', width=2),
            marker=dict(symbol='square', size=8),
        ))
        fig_elbow.add_vline(x=best_k, line_dash='dash', line_color='tomato',
                             annotation_text=f'Silhouette best k = {best_k}',
                             annotation_position='top right')
        fig_elbow.update_layout(
            title='Elbow Method (Inertia / WCSS)',
            xaxis_title='Number of clusters (k)', yaxis_title='Inertia (WCSS)',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    st.info(
        f"✅ **Silhouette Score** selects **k = {best_k}** as the optimal number of clusters."
    )
    sil_df = pd.DataFrame({
        'k': k_vals,
        'Silhouette Score': [f'{s:.4f}' for s in sil_scores],
        '': [' ← best' if v == best_k else '' for v in k_vals],
    })
    st.dataframe(sil_df, hide_index=True, use_container_width=False)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — K-Means & PCA
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("K-Means Clustering Analysis")

    k = st.sidebar.slider(
        "Number of Clusters (k) for K-Means", min_value=2, max_value=8, value=2, step=1
    )

    # Build a local copy — never mutate the cached singleton
    df2 = df.copy()
    df2['Cluster'] = labels_dict[k].astype(str)    # pre-computed, instant
    df2['PCA1']    = pca_components[:, 0]            # pre-computed, instant
    df2['PCA2']    = pca_components[:, 1]

    sample_size = st.sidebar.slider("Sample size for PCA Plot", 500, 10_000, 2_000, 500)
    df_sample = df2.sample(n=min(sample_size, len(df2)), random_state=42)

    fig_scatter = px.scatter(
        df_sample, x='PCA1', y='PCA2', color='Cluster',
        hover_data=['track_name', 'track_artist', 'playlist_genre'],
        title=f'2D PCA Visualization (k={k})',
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_scatter.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### Cluster Profiles Heatmap")
    cluster_means = df2.groupby('Cluster')[X.columns].mean()
    scaler_means  = StandardScaler()
    cluster_means_scaled = pd.DataFrame(
        scaler_means.fit_transform(cluster_means),
        columns=cluster_means.columns,
        index=cluster_means.index,
    )
    fig_heat = px.imshow(
        cluster_means_scaled.T, aspect='auto',
        color_continuous_scale='Viridis',
        title='Relative Feature Importance per Cluster',
    )
    fig_heat.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Hierarchical Clustering
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Hierarchical Clustering")
    st.markdown(
        "Dendrogram generated using **Agglomerative Clustering** with distance thresholds."
    )
    # hc_model loaded from disk — no wait
    fig_dendro = get_dendrogram_fig(hc_model, truncate_mode='level', p=10)
    st.pyplot(fig_dendro)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Playlist Recommender
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("🎵 Playlist Recommender")
    st.markdown(
        "Select up to **5 of your favourite songs** from the dataset.  \n"
        "The app identifies which cluster they belong to and recommends a "
        "playlist from the best-matching cluster."
    )

    k_rec = st.slider(
        "Number of clusters (k) to use for recommendations",
        min_value=2, max_value=8, value=4, step=1,
        key='k_recommender',
    )

    # Build "Song Name — Artist" display labels from unique (name, artist) pairs
    _pairs = (
        df[['track_name', 'track_artist']]
        .dropna()
        .drop_duplicates()
        .sort_values('track_name')
    )
    _pairs['label'] = _pairs['track_name'] + ' — ' + _pairs['track_artist']
    _pairs = _pairs.drop_duplicates(subset='label')          # guard against exact dupes
    _label_to_track = dict(zip(_pairs['label'], _pairs['track_name']))
    all_labels = _pairs['label'].tolist()

    selected_labels = st.multiselect(
        "🔍 Search and select up to 5 songs:",
        options=all_labels,
        max_selections=5,
        placeholder="Type a song name or artist…",
    )
    # Map display labels back to plain track_names for the recommendation engine
    selected_songs = [_label_to_track[lbl] for lbl in selected_labels]

    if st.button("🎧 Find My Playlist", type="primary"):
        if not selected_songs:
            st.warning("Please select at least one song first.")
        else:
            # Build cluster-labelled df using pre-computed labels — no KMeans run
            df_clustered = df.copy()
            df_clustered['Cluster'] = labels_dict[k_rec]

            results = recommend_playlists(df_clustered, selected_songs, k_rec)

            if not results or results[0]['count'] == 0:
                st.error(
                    "Could not find any of the selected songs in the dataset. "
                    "Please try different songs."
                )
            else:
                # Cluster match bar chart
                st.markdown("### 📊 Cluster Match Scores")
                score_df = pd.DataFrame([
                    {
                        'Cluster':      f"Cluster {r['cluster']}",
                        'Match (%)':    round(r['score'] * 100, 1),
                        'Songs Found':  r['count'],
                        'Cluster Size': r['total_tracks'],
                    }
                    for r in results
                ])
                fig_scores = px.bar(
                    score_df, x='Cluster', y='Match (%)',
                    color='Match (%)', color_continuous_scale='Viridis',
                    title='Cluster Match Scores for Your Selected Songs',
                    text='Match (%)',
                )
                fig_scores.update_traces(texttemplate='%{text}%', textposition='outside')
                fig_scores.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    coloraxis_showscale=False, yaxis_range=[0, 115],
                )
                st.plotly_chart(fig_scores, use_container_width=True)

                # Best cluster
                best = results[0]
                cid  = best['cluster']
                st.markdown(f"### 🏆 Best Match: Cluster {cid}")
                st.markdown(
                    f"**{best['count']}** of your **{len(selected_songs)}** selected "
                    f"songs ({round(best['score']*100, 1)}%) belong to "
                    f"**Cluster {cid}**, which contains **{best['total_tracks']:,}** "
                    "songs in total."
                )

                cluster_tracks = df_clustered[df_clustered['Cluster'] == cid]
                col_pie, col_table = st.columns([1, 2])

                with col_pie:
                    st.markdown("#### Genre Breakdown")
                    if 'playlist_genre' in cluster_tracks.columns:
                        gd = cluster_tracks['playlist_genre'].value_counts().reset_index()
                        gd.columns = ['Genre', 'Count']
                        fig_pie = px.pie(
                            gd, names='Genre', values='Count',
                            color_discrete_sequence=px.colors.qualitative.Set2,
                            title=f'Cluster {cid} — Genre Mix',
                        )
                        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_pie, use_container_width=True)

                with col_table:
                    st.markdown("#### Sample Tracks from Your Playlist")
                    display_cols = [
                        c for c in
                        ['track_name', 'track_artist', 'playlist_genre', 'track_popularity']
                        if c in cluster_tracks.columns
                    ]
                    sample = (
                        cluster_tracks[display_cols]
                        .sample(min(25, len(cluster_tracks)), random_state=42)
                        .reset_index(drop=True)
                    )
                    sample.index += 1
                    st.dataframe(sample, use_container_width=True)
