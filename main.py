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

        Args:
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

    def slice_data(self, data: np.ndarray) -> np.ndarray:
        """
        Slice the data array with user's column

        Args:
            data (np.ndarray): The data array to slice

        Returns:
            np.ndarray: The sliced data array
        """
        return data[:, self.col_idx].astype(float)


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

    # Get cmap and define anonymous function to rescale data to [0,1]
    # Get list of colors from cmap and scaled data
    cmap = plt.get_cmap("tab20")
    rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
    colors = list(map(cmap, rescale(data)))

    plt.pie(data, labels=data_labels, autopct="%1.1f%%", pctdistance=0.8, colors=colors)
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
    # Uses Pearson correlation coefficient

    # Collect sliced data for users
    user1_data = user1.slice_data(data)
    user2_data = user2.slice_data(data)

    # Normalize user data
    user1_data /= np.max(user1_data)
    user2_data /= np.max(user2_data)

    # Calculate correlation coefficient matrix
    cor_matrix = np.corrcoef(user1_data, user2_data)

    diffs = abs(user1_data - user2_data)
    filter_mask = np.concatenate((np.argwhere(user1_data != 0), np.argwhere(user2_data != 0)))
    diffs = diffs[filter_mask]  # Filter out songs with no scrobbles
    max_cor = np.argmin(np.abs(diffs))
    min_cor = np.argmax(np.abs(diffs))

    print(f"{user1.name} and {user2.name} have a correlation of {cor_matrix[0, 1]:.4f}")
    print(
        f"{data[max_cor, 0]} has the highest correlation for {user1.name} and {user2.name}",
        f"with a difference of {diffs[max_cor][0]:.4f}",
        f"({user1.name}: {user1_data[max_cor]:.4f}, {user2.name}: {user2_data[max_cor]:.4f})",
    )
    print(
        f"{data[min_cor, 0]} has the lowest correlation for {user1.name} and {user2.name}",
        f"with a difference of {diffs[min_cor][0]:.4f}",
        f"({user1.name}: {user1_data[min_cor]:.4f}, {user2.name}: {user2_data[min_cor]:.4f})",
    )

    plot_similarity(user1, user2, user1_data, user2_data)


def plot_similarity(user1: User, user2: User, user1_data: np.ndarray, user2_data: np.ndarray):
    # Filter points where one user has 0 scrobbles
    # Truncate floats to better group points
    points = np.array(
        [
            (truncate(x, 3), truncate(y, 3))
            for x, y in zip(user1_data, user2_data)
            if truncate(x, 3) != 0 and truncate(y, 3) != 0
        ]
    )
    user1_data = np.array([p[0] for p in points])
    user2_data = np.array([p[1] for p in points])

    # Count number of points at each point and assign color value based on count
    colors = np.array([np.count_nonzero(points == x) for x in points])

    plt.scatter(user1_data, user2_data, c=colors, cmap="viridis")
    # Plot y=x line to show similarity
    plt.plot(
        [i / 10 for i in range(-1, 12)],
        [i / 10 for i in range(-1, 12)],
        "k",
        linewidth=1,
        zorder=0,
    )
    plt.ylim(bottom=-0.05, top=1.05)
    plt.xlim(left=-0.05, right=1.05)
    plt.title(f"Correlation Between {user1.name} and {user2.name}")
    plt.xlabel(f"{user1.name}'s Scrobbles")
    plt.ylabel(f"{user2.name}'s Scrobbles")
    plt.show()


def discography_depth(user: User, data: np.ndarray):
    # Must be album_data
    # Most album listens per artist

    # Filter data to where user has at least 1 scrobble of an album
    data = data[user.album_data > 1]
    # Get a set of unique artist names
    artists = remove_duplicates(data[:, 1].astype(str))  # Index 1 is artist column
    # Get the indices of all albums for each unique artist
    artist_counts = [np.argwhere(data[:, 1] == artist) for artist in artists]
    # Length of indices list will be number of albums for that artist
    album_counts = [len(artist_count) for artist_count in artist_counts]
    # Get max number of albums and index for the max
    most_albums = np.max(album_counts)
    most_albums_idx = np.argmax(album_counts)
    print(f"{user.name} has gone deepest on {artists[most_albums_idx]} with {most_albums} albums")
    plot_discography_depth(user, np.array(album_counts, dtype=int), np.array(artists, dtype=str))


def plot_discography_depth(user: User, album_counts: np.ndarray, artists: np.ndarray, n: int = 10):
    """
    Plot the top n artists with the most albums

    Args:
        user (User): The user object
        album_counts (list[int]): The number of albums for each artist
        artists: (list[str]): The artist names
        n (int, optional): The number of artists to plot. Defaults to 10.
    """
    partition = np.argpartition(album_counts, -n)  # Partition so top n are sorted to end
    top_n = partition[-n:]
    top_n = top_n[np.argsort(album_counts[top_n])][::-1]  # Sort top n

    # Limit artist names to certain length
    artists = np.array(
        [artist[:15] + "..." if len(artist) > 15 else artist for artist in artists], dtype=str
    )

    data = album_counts[top_n]
    data_labels = artists[top_n]

    # Get cmap and define anonymous function to rescale data to [0,1]
    cmap = plt.get_cmap("viridis")
    rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))

    plt.bar(data_labels, data, tick_label=data_labels, color=cmap(rescale(data)))
    plt.xticks(rotation=80)
    plt.xlabel("Artists")
    plt.ylabel("Albums Listened To")
    plt.title(f"{user.name}'s Deepest Discography Dives")
    plt.grid(axis="y", linewidth=0.4)
    plt.tight_layout()
    plt.show()


def remove_duplicates(data: list):
    """Remove duplicate entries from a list, maintaining order

    Args:
        data (list): The data to filter

    Returns:
        list: Filtered data
    """
    seen = []
    for item in data:
        if item not in seen:
            seen.append(item)
    return seen


def truncate(number: float, digits: int):
    """Truncate a number to a certain number of digits

    Args:
        number (float): The number to truncate
        digits (int): The number of digits to keep

    Returns:
        float: The truncated number
    """
    return int(number * 10 ** digits) / 10 ** digits


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
    # average_duration(jon, track_data)
    # discography_depth(jon, album_data)
    similarity(jon, matt, track_data)


if __name__ == "__main__":
    main()
