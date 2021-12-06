# JONATHAN BREIDFJORD
# Pandas used to import data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D  # For custom legends


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
    """Calculates the user's mainstream score.\n
    Uses a weighted average of the artist's total number of plays where the weights
    are the percentage of the user's total scrobbles for all artists.

    Args:
        user (User): The user to analyze
        data (np.ndarray): The data to analyze (must be artist data)
    """
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
        f"\n{user.name}'s mainstream score is {np.sum(artist_contributions)}\n",
        f"{user.name}'s most mainstream artist is ",
        f"'{artist_names[max_contrib_idx]}' with a contribution of ",
        f"{max_contrib:.2f} ({artist_contributions_pct[max_contrib_idx] * 100:.2f}%)",
        sep="",
    )
    plot_mainstream(user, artist_contributions_pct, artist_names)


def plot_mainstream(
    user: User, artist_contributions: np.ndarray, artist_names: list[str], n: int = 9
):
    """Plot the top n artists with the highest contribution to the user's mainstream score

    Args:
        user (User): The user to analyze
        artist_contributions (np.ndarray): The artist contributions to the user's mainstream score
        artist_names (list[str]): The names of the artists, must be sorted with contributions
        n (int, optional): The number of artists to plot. Defaults to 9.
    """
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
    """Calculate the average duration of a user's tracks.

    Args:
        user (User): The user to analyze
        data (np.ndarray): The dataset to analyze (must be track data)
    """
    duration_data = data[:, 3].astype(float)  # Index 3 is duration of track
    # Filter duration_data to songs with at least 1 scrobble by user
    duration_data = duration_data[user.track_data > 0]

    print(f"\n{user.name} listens to {np.mean(duration_data):.2f} second long songs on average")


def similarity(user1: User, user2: User, data: np.ndarray):
    """Calculates correlation between two user's tastes based on the given dataset.
    Uses the Pearson correlation coefficient.

    Args:
        user1 (User): First user to analyze
        user2 (User): Second user to analyze, must be a different user
        data (np.ndarray): The dataset to analyze
    """
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

    print(f"\n{user1.name} and {user2.name} have a correlation of {cor_matrix[0, 1]:.4f}")
    print(
        f"'{data[max_cor, 0]}' has the highest correlation",
        f"with a difference of {diffs[max_cor][0]:.4f}",
        f"({user1.name}: {user1_data[max_cor]:.4f}, {user2.name}: {user2_data[max_cor]:.4f})",
    )
    print(
        f"'{data[min_cor, 0]}' has the lowest correlation",
        f"with a difference of {diffs[min_cor][0]:.4f}",
        f"({user1.name}: {user1_data[min_cor]:.4f}, {user2.name}: {user2_data[min_cor]:.4f})",
    )

    plot_similarity(user1, user2, user1_data, user2_data)


def plot_similarity(user1: User, user2: User, user1_data: np.ndarray, user2_data: np.ndarray):
    """Scatter plot showing correlation between two users.\n
    Data will be filtered to exclude points where one user has no scrobbles.\n
    Points are grouped and coloured based on how many points are in the grouping.

    Args:
        user1 (User): First user to analyze, will be x data
        user2 (User): Second user to analyze, will be y data
        user1_data (np.ndarray): Data for user1
        user2_data (np.ndarray): Data for user2
    """
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

    # Set up points for custom legend
    cmap = plt.get_cmap("viridis")
    legend_elems = [
        Line2D([0], [0], marker="o", markerfacecolor=cmap(0.1), color="w", label="Single point"),
        Line2D([0], [0], marker="o", markerfacecolor=cmap(0.5), color="w", label="Some points"),
        Line2D([0], [0], marker="o", markerfacecolor=cmap(1.0), color="w", label="Many points"),
    ]

    plt.scatter(user1_data, user2_data, c=colors, cmap=cmap, alpha=0.75)
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
    plt.legend(handles=legend_elems)
    plt.show()


def discography_depth(user: User, data: np.ndarray):
    """Calculates the number of unique albums scrobbled and ranks the artists

    Args:
        user (User): The user object to analyze
        data (np.ndarray): The data to analyze (must be album data)
    """
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
    print(
        f"\n{user.name} has gone deepest on '{artists[most_albums_idx]}' with {most_albums} albums"
    )
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
    plt.ylabel("Albums Scrobbled")
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


def menu(names: list[str]) -> tuple[int, tuple[int], str]:
    """Print the menus and get user selections

    Args:
        names (list[str]): The names of the possible users to analyze

    Returns:
        tuple[int, tuple[int], str]: Analysis option, index of user(s) to analyze, dataset to analyze
    """
    # Dict to map menu options to possible data types for analysis
    opt_data_map = {
        1: ["Tracks", "Albums", "Artists"],
        2: ["Albums"],
        3: ["Artists"],
        4: ["Tracks"],
    }

    print(f"{' Analysis Options ':-^40}")
    print("1) Compare two users")
    print("2) Discography depth")
    print("3) Mainstream score")
    print("4) Average track length")
    print("0) Quit")
    while (analysis_opt := input("\nEnter your numeric choice: ")) not in ["1", "2", "3", "4", "0"]:
        print("Invalid choice, please enter a numeric option from the menu")
    else:
        analysis_opt = int(analysis_opt)
    if analysis_opt == 0:
        return analysis_opt, None, None  # Return type must be tuple

    if len((data_types := opt_data_map[analysis_opt])) > 1:
        print(f"\n{' Data Types ':-^40}")
        for i, data_type in enumerate(data_types, start=1):
            print(f"{i}) {data_type}")
        print("0) Quit")
        while (data_opt := input("\nEnter choice by name or number: ").lower()) not in [
            str(i) for i in range(len(data_types) + 1)
        ] + [dt.lower() for dt in data_types] + ["quit"]:
            print("Invalid choice, please enter an option from the menu")
        else:
            data_opt = int(data_opt) - 1 if data_opt.isdigit() else data_opt
        if data_opt == -1 or data_opt == "quit":
            return None, None, None
        elif isinstance(data_opt, int):
            data_opt = data_types[data_opt]
    else:
        data_opt = opt_data_map[analysis_opt][0]

    print(f"\n{' User Selection ':-^40}")
    for i, name in enumerate(names, start=1):
        print(f"{i}) {name}")
    print("0) Quit")
    while (user_opt := input("\nEnter choice by name or number: ")) not in [
        str(i) for i in range(len(names) + 1)
    ]:
        print("Invalid choice, please enter a numeric option from the menu")
    else:
        user_opt = int(user_opt) - 1
    if user_opt == -1:
        return analysis_opt, (user_opt,), data_opt  # Return type must be tuple

    if analysis_opt == 1:  # Take 2nd user input for comparison if selected
        print(f"\n{' 2nd User Selection ':-^40}")
        for i, name in enumerate(names, start=1):
            print(f"{i}) {name}")
        print("0) Quit")
        while (user2_opt := input("\nEnter choice by name or number: ")) not in [
            str(i) for i in range(len(names) + 1) if i != user_opt + 1
        ]:
            print("Invalid choice, please enter a numeric option from the menu")
        else:
            user2_opt = int(user2_opt) - 1
        if user2_opt == -1:
            return None, (user2_opt,)
        return analysis_opt, (user_opt, user2_opt), data_opt

    return analysis_opt, (user_opt,), data_opt


def main():
    # Import data
    artist_data = np.asarray(pd.read_csv("artist_data.csv", dtype=str))
    album_data = np.asarray(pd.read_csv("album_data.csv", dtype=str))
    track_data = np.asarray(pd.read_csv("track_data.csv", dtype=str))
    # Get names from column headers
    names = [name.title() for name in pd.read_csv("artist_data.csv", dtype=str).columns[-4:]]

    # Map menu options to analysis functions
    analysis_map = {1: similarity, 2: discography_depth, 3: mainstream, 4: average_duration}

    # Get user input
    analysis_opt, analysis_args, data_opt = menu(names)
    if analysis_opt == 0 or data_opt == None or -1 in analysis_args:
        print("\nGoodbye!")
        return

    # Get data type based on selected data option
    data_type = (
        artist_data if data_opt == "Artists" else album_data if data_opt == "Albums" else track_data
    )

    # Convert input to User objects
    analysis_args = [User(i - 4, names[i]) for i in analysis_args]
    for user in analysis_args:  # Add data to each User
        user.collect_data(artist_data, album_data, track_data)

    analysis_map[analysis_opt](*analysis_args, data_type)


if __name__ == "__main__":
    main()
