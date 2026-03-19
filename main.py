import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("spotify_songs.csv", index_col=1)
data.drop(["track_id", "track_album_id", "playlist_id", "playlist_name", "playlist_genre", "playlist_subgenre"], inplace=True, axis=1)
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
# Normalize year-only release dates like '2012' to '2012-01-01'
data['track_album_release_date'] = data['track_album_release_date'].astype(str).str.strip()
mask = data['track_album_release_date'].str.match(r'^\d{4}$')
data.loc[mask, 'track_album_release_date'] = data.loc[mask, 'track_album_release_date'] + '-01-01'
# Now parse all dates using YYYY-MM-DD format
pd.to_datetime(data['track_album_release_date'], format='mixed')
