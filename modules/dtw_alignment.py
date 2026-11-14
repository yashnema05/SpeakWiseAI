
import numpy as np
from scipy.spatial.distance import cdist


def align_features(reference_features, participant_features):
    """
    Align reference and participant MFCC feature sequences using
    Dynamic Time Warping (DTW).

    Both inputs should have shape:
    (number_of_features, number_of_time_frames)
    """

    reference = np.asarray(reference_features, dtype=float)
    participant = np.asarray(participant_features, dtype=float)

    if reference.ndim != 2 or participant.ndim != 2:
        raise ValueError("Both feature inputs must be 2D arrays.")

    if reference.shape[0] != participant.shape[0]:
        raise ValueError(
            "Reference and participant must have the same number "
            "of feature dimensions."
        )

    if reference.shape[1] == 0 or participant.shape[1] == 0:
        raise ValueError("Feature sequences cannot be empty.")

    # Each column represents one time frame.
    ref_frames = reference.T
    participant_frames = participant.T

    # Pairwise Euclidean distances between frames.
    distances = cdist(ref_frames, participant_frames, metric="euclidean")

    n, m = distances.shape

    # Dynamic programming cost matrix.
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            previous_cost = min(
                cost[i - 1, j],
                cost[i, j - 1],
                cost[i - 1, j - 1]
            )

            cost[i, j] = distances[i - 1, j - 1] + previous_cost

    # Trace the lowest-cost path backwards.
    i, j = n, m
    path = []

    while i > 0 and j > 0:
        path.append((i - 1, j - 1))

        options = [
            cost[i - 1, j - 1],
            cost[i - 1, j],
            cost[i, j - 1]
        ]

        move = int(np.argmin(options))

        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1

    while i > 0:
        i -= 1
        path.append((i, 0))

    while j > 0:
        j -= 1
        path.append((0, j))

    path.reverse()

    normalized_cost = cost[n, m] / max(len(path), 1)

    return {
        "path": path,
        "distance_matrix": distances,
        "normalized_cost": float(normalized_cost)
    }
