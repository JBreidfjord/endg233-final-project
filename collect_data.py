# %%
import os
import time

import pandas as pd
import pylast
from dotenv import load_dotenv
from tqdm.auto import tqdm

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

users = ["JBreidfjord", "IcoBeltran", "TheRudeDuck", "superfrat"]
artist_data = {}
album_data = {}
track_data = {}

# %%
for username in users:
    user = network.get_user(username)

    artists = user.get_top_artists(limit=1000)
    for item, weight in tqdm(artists, desc=f"{username} artists"):
        # Only grab artist data if the artist doesn't exist
        # Otherwise only add scrobbles from user
        try:
            artist = item.get_name(properly_capitalized=True)
        except pylast.MalformedResponseError:
            print(f"Bad response for {artist} name")
            artist = item.name
        if artist_dict := artist_data.get(artist):
            artist_dict[username + "_scrobbles"] = weight
        else:
            artist_dict = {u + "_scrobbles": 0 for u in users}
            artist_dict[username + "_scrobbles"] = weight
            try:
                artist_dict["total_plays"] = item.get_playcount()
            except pylast.MalformedResponseError:
                print(f"Bad response for {artist} playcount")
                artist_dict["total_plays"] = 0
            artist_data[artist] = artist_dict
        time.sleep(0.1)  # Be nice to Last.fm

artist_df = pd.DataFrame(artist_data.values(), index=artist_data.keys())
artist_df.index.name = "artist"
cols = artist_df.columns.tolist()
cols = cols[-1:] + cols[:-1]  # Rearrange columns
artist_df = artist_df[cols]
artist_df.to_csv("artist_data.csv")

# %%
for username in users:
    user = network.get_user(username)
    albums = user.get_top_albums(limit=1000)
    for item, weight in tqdm(albums, desc=f"{username} albums"):
        # Only grab album data if the artist doesn't exist
        # Otherwise only add scrobbles from user
        if album_dict := album_data.get(item.title):
            album_dict[username + "_scrobbles"] = weight
        else:
            album_dict = {u + "_scrobbles": 0 for u in users}
            album_dict[username + "_scrobbles"] = weight
            try:
                album_dict["artist"] = item.artist.get_name(properly_capitalized=True)
            except pylast.MalformedResponseError:
                print(f"Bad response for {item.artist} name")
                album_dict["artist"] = item.artist
            album_data[item.title] = album_dict
        time.sleep(0.1)  # Be nice to Last.fm

album_df = pd.DataFrame(album_data.values(), index=album_data.keys())
album_df.index.name = "album"
cols = album_df.columns.tolist()
cols = cols[-1:] + cols[:-1]  # Rearrange columns
album_df = album_df[cols]
album_df.to_csv("album_data.csv")

# %%
for username in users:
    user = network.get_user(username)
    tracks = user.get_top_tracks(limit=1000)
    for item, weight in tqdm(tracks, desc=f"{username} tracks"):
        # Only grab track data if the artist doesn't exist
        # Otherwise only add scrobbles from user
        track = item.get_correction()
        if track_dict := track_data.get(track):
            track_dict[username + "_scrobbles"] = weight
        else:
            track_dict = {u + "_scrobbles": 0 for u in users}
            track_dict[username + "_scrobbles"] = weight
            try:
                track_dict["artist"] = item.artist.get_name(properly_capitalized=True)
            except pylast.MalformedResponseError:
                print(f"Bad response for {item.artist} name")
                track_dict["artist"] = item.artist
            try:
                track_dict["album"] = item.get_album().title
            except AttributeError:
                print(f"No album for {item.artist} - {track}")
                track_dict["album"] = "Unknown"
            except pylast.MalformedResponseError:
                print(f"Bad response for {item.artist} - {track}")
                track_dict["album"] = "Unknown"
            track_dict["duration"] = item.get_duration()
            track_data[track] = track_dict
        time.sleep(0.1)  # Be nice to Last.fm

track_df = pd.DataFrame(track_data.values(), index=track_data.keys())
track_df.index.name = "track"
cols = track_df.columns.tolist()
cols = cols[-3:] + cols[:-3]  # Rearrange columns
cols[0], cols[1] = cols[1], cols[0]  # Rearrange columns
track_df = track_df[cols]
track_df["duration"] //= 1000  # Convert to seconds
track_df.to_csv("track_data.csv")

# %%
