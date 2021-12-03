# JONATHAN BREIDFJORD
# Pandas used to import data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class User:
    """User class representing a user with a column index for the data arrays"""

    def __init__(self, col_idx: int, name: str):
        assert -4 <= col_idx < 0, "Column index must be between -4 and 0"
        self.col_idx = col_idx
        self.name = name

        # Must be set with the collect_data method
        self.artist_data = None
        self.album_data = None
        self.track_data = None

    def collect_data(self, artist_data: np.ndarray, album_data: np.ndarray, track_data: np.ndarray):
        """
        Collect user data from the data arrays and store in attributes

        Params:
            artist_data (np.ndarray): The data array containing artist data
            album_data (np.ndarray): The data array containing album data
            track_data (np.ndarray): The data array containing track data

        Returns:
            None
        """
        # Cast each slice to float
        self.artist_data = artist_data[:, self.col_idx].astype(float)
        self.album_data = album_data[:, self.col_idx].astype(float)
        self.track_data = track_data[:, self.col_idx].astype(float)


def mainstream(user: User, data: np.ndarray):
    # Must be artist_data
    total_scrobbles = np.sum(user.artist_data)
    play_data = data[:, 1].astype(float)  # Index 1 is total_plays column
    artist_names = data[:, 0].astype(str)  # Index 0 is artist column
    artist_contributions = []
    for plays, scrobbles in zip(play_data, user.artist_data):
        scrobble_fraction = scrobbles / total_scrobbles
        artist_weight = plays * scrobble_fraction
        artist_contributions.append(artist_weight)

    artist_contributions = np.array(artist_contributions)  # Convert to np array
    # Normalize contributions to get a percentage
    artist_contributions_pct = artist_contributions / np.sum(artist_contributions)

    max_contrib = np.max(artist_contributions)
    max_contrib_idx = np.argmax(artist_contributions)
    print(
        f"{user.name}'s mainstream score is {np.sum(artist_contributions)}\n",
        f"{user.name}'s most mainstream artist is ",
        f"{artist_names[max_contrib_idx]} with a contribution of ",
        f"{max_contrib:.2f} ({artist_contributions_pct[max_contrib_idx] * 100:.2f}%)",
        sep="",
    )
    plot_mainstream(user, artist_contributions_pct, artist_names)


def plot_mainstream(
    user: User, artist_contributions: np.ndarray, artist_names: list[str], n: int = 9
):
    partition = np.argpartition(artist_contributions, -n)  # Partition so top n are sorted to end
    top_n = partition[-n:]
    other_contribs = np.sum(artist_contributions[partition[:-n]])
    top_n = top_n[np.argsort(artist_contributions[top_n])][::-1]  # Sort top n

    data = np.concatenate((artist_contributions[top_n], (other_contribs,)))
    data_labels = np.concatenate((artist_names[top_n], ("Other",)))

    plt.pie(data, labels=data_labels, autopct="%1.1f%%", pctdistance=0.8)
    plt.title(f"{user.name}'s Most Mainstream Artists")
    plt.show()


def average_duration(user: User, data: np.ndarray):
    # Must be track_data
    duration_data = data[:, 3].astype(float)  # Index 3 is duration of track
    # Filter duration_data to songs with at least 1 scrobble by user
    duration_data = duration_data[user.track_data > 0]

    print(f"{user.name} listens to {np.mean(duration_data):.2f} second long songs on average")


def similarity(user1: User, user2: User, data: np.ndarray):
    # Similarity in tastes between users
    # Maybe similar to mainstream where it takes a weighted average of artists
    # Then gives a score based on how close each contrib is
    # Could then use mean to calculate average difference
    # E.g. user1: [0.4, 0.5, 0.1] user2: [0.2, 0.5, 0.3] = diffs: [0.2, 0.0, -0.2]
    ...


def discography_depth(user: User, data: np.ndarray):
    # Must be album_data
    # Most album listens per artist
    data = data[user.album_data > 1]


def main():
    # Import data
    artist_data = np.asarray(pd.read_csv("artist_data.csv", dtype=str))
    album_data = np.asarray(pd.read_csv("album_data.csv", dtype=str))
    track_data = np.asarray(pd.read_csv("track_data.csv", dtype=str))

    print("Artist Data:", "\n", artist_data, "\n")
    print("Album Data:", "\n", album_data, "\n")
    print("Track Data:", "\n", track_data, "\n")

    jon = User(-4, "Jon")
    jon.collect_data(artist_data, album_data, track_data)
    ico = User(-3, "Ico")
    ico.collect_data(artist_data, album_data, track_data)
    matt = User(-2, "Matt")
    matt.collect_data(artist_data, album_data, track_data)
    austin = User(-1, "Austin")
    austin.collect_data(artist_data, album_data, track_data)

    # mainstream(jon, artist_data)
    average_duration(jon, track_data)
    discography_depth(jon, album_data)


if __name__ == "__main__":
    main()
