# %%
import os

import pandas as pd
import pylast
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["API_KEY"]
API_SECRET = os.environ["API_SECRET"]

# Only required for write operations
USERNAME = os.environ["USERNAME"]
PASSWORD_HASH = pylast.md5(os.environ["PASSWORD"])

# %%
network = pylast.LastFMNetwork(
    api_key=API_KEY,
    api_secret=API_SECRET,
    username=USERNAME,
    password_hash=PASSWORD_HASH,
)

# %%
all_artist_data = []
all_album_data = []
all_track_data = []
for username in ["JBreidfjord", "IcoBeltran", "TheRudeDuck", "superfrat"]:
    user = network.get_user(username)

    artists = user.get_top_artists(limit=1000)
    artist_data = [
        {"artist": artist.item, "scrobbles": artist.weight, "user": username} for artist in artists
    ]
    all_artist_data.extend(artist_data)

    albums = user.get_top_albums(limit=1000)
    album_data = [
        {"album": album.item, "scrobbles": album.weight, "user": username} for album in albums
    ]
    all_album_data.extend(album_data)

    tracks = user.get_top_tracks(limit=1000)
    track_data = [
        {"track": track.item, "scrobbles": track.weight, "user": username} for track in tracks
    ]
    all_track_data.extend(track_data)

# %%
artist_df = pd.DataFrame(all_artist_data)
artist_df.to_csv("artist_data.csv", index=False)

album_df = pd.DataFrame(all_album_data)
album_df.to_csv("album_data.csv", index=False)

track_df = pd.DataFrame(all_track_data)
track_df.to_csv("track_data.csv", index=False)

# %%
