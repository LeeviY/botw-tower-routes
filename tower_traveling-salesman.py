from dataclasses import dataclass
from itertools import permutations
import math
from time import time
import numpy as np


@dataclass
class Tower:
    name: str
    x: int
    y: int
    h: int


def print_path(path: list[Tower]):
    print(" -> ".join(towers[i].name for i in path))


def brute_force_tsp(distance_matrix: list[list[int]]):
    best_path = None
    min_distance = float("inf")
    t = list(range(len(towers)))

    checked = 0
    start = time()
    for perm in permutations(t):
        total_distance = 0

        for i in range(len(perm) - 1):
            total_distance += distance_matrix[perm[i], perm[i + 1]]
            if total_distance > min_distance:
                break

        if total_distance < min_distance:
            min_distance = total_distance
            best_path = perm
            print_path(best_path)
            print(
                f"Min Distance: {min_distance}, Checked: {checked}, Took: {round(time() - start, 2)} s"
            )

        checked += 1

    return best_path, min_distance


towers: list[Tower] = [
    Tower("Akkala", 3308, -1500, 0),
    Tower("Central", -788, 441, 0),
    Tower("DuelingPeaks", 1017, 1714, 0),
    Tower("Eldin", 2174, -1557, 0),
    Tower("Faron", 1331, 3274, 0),
    Tower("Gerudo", -3666, 1829, 0),
    Tower("GreatPlateau", -648, 1833, 0),
    Tower("Hateno", 2736, 2135, 0),
    Tower("Hebra", -2173, -2034, 0),
    Tower("Lake", -32, 2963, 0),
    Tower("Lanayru", 2258, -109, 0),
    Tower("Ridgeland", -1755, -775, 0),
    Tower("Tabantha", -3614, -990, 0),
    Tower("Wasteland", -2306, 2439, 0),
    Tower("Woodland", 884, -1607, 0),
]


tower_positions = np.array([(tower.x, tower.y) for tower in towers])

distance_matrix = np.round(
    np.sqrt(np.sum((tower_positions[:, None, :] - tower_positions) ** 2, axis=2))
).astype(int)

brute_force_tsp(distance_matrix)
