from dataclasses import dataclass
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


@dataclass
class Tower:
    name: str
    x: int
    y: int
    h: int


calls = 0


def held_karp_tsp(distance_matrix: np.ndarray):
    num_towers = distance_matrix.shape[0]

    # Initialize the memoization table and path table
    memo = np.full((1 << num_towers, num_towers), -1)
    path_table = np.zeros((1 << num_towers, num_towers), dtype=int)

    # Helper function for the recursive Held-Karp algorithm
    def held_karp_helper(mask, current):
        global calls
        calls += 1
        if mask == (1 << num_towers) - 1:
            return distance_matrix[current, 0]

        if memo[mask, current] != -1:
            return memo[mask, current]

        min_distance = float("inf")
        next_tower_at_min_distance = None

        for next_tower in range(num_towers):
            if (mask & (1 << next_tower)) == 0:
                new_mask = mask | (1 << next_tower)
                distance = distance_matrix[current, next_tower] + held_karp_helper(
                    new_mask, next_tower
                )

                if distance < min_distance:
                    min_distance = distance
                    next_tower_at_min_distance = next_tower

        memo[mask, current] = min_distance
        path_table[mask, current] = next_tower_at_min_distance

        return min_distance

    # Start the Held-Karp algorithm from the first tower (index 0)
    min_distance = held_karp_helper(1, 0)

    # Reconstruct the shortest path
    path = []
    mask = 1
    current = 0

    for _ in range(num_towers - 1):
        next_tower = path_table[mask, current]
        path.append(next_tower)
        mask |= 1 << next_tower
        current = next_tower

    # Add the starting tower to complete the path
    path.append(0)

    return min_distance, path


towers: list[Tower] = [
    Tower("Akkala", 3308, -1500, 524),
    Tower("Central", -788, 441, 128),
    Tower("DuelingPeaks", 1017, 1714, 114),
    Tower("Eldin", 2174, -1557, 439),
    Tower("Faron", 1331, 3274, 200),
    Tower("Gerudo", -3666, 1829, 401),
    Tower("GreatPlateau", -560, 1695, 177),
    Tower("Hateno", 2736, 2135, 266),
    Tower("Hebra", -2173, -2034, 459),
    Tower("Lake", -32, 2963, 211),
    Tower("Lanayru", 2258, -109, 241),
    Tower("Ridgeland", -1755, -775, 259),
    Tower("Tabantha", -3614, -990, 375),
    Tower("Wasteland", -2306, 2439, 460),
    Tower("Woodland", 884, -1607, 280),
]


Akkala = 0
Central = 1
DuelingPeaks = 2
Eldin = 3
Faron = 4
Gerudo = 5
GreatPlateau = 6
Hateno = 7
Hebra = 8
Lake = 9
Lanayru = 10
Ridgeland = 11
Tabantha = 12
Wasteland = 13
Woodland = 14


tower_positions = np.array([(tower.x, tower.y) for tower in towers])

distance_matrix = np.zeros((len(towers), len(towers)))

for i in range(len(towers)):
    for j in range(len(towers)):
        distance = np.sqrt(
            (towers[i].x - towers[j].x) ** 2 + (towers[i].y - towers[j].y) ** 2
        )
        height_diff = max(0, towers[i].h - towers[j].h)
        distance_matrix[i, j] = distance / 100 + height_diff / 3

distance_matrix[DuelingPeaks, Faron] = 99999999
distance_matrix[DuelingPeaks, Hateno] = 99999999

distance_matrix[Faron, DuelingPeaks] = 99999999
distance_matrix[Faron, Hateno] = 99999999
distance_matrix[Faron, Lanayru] = 99999999

distance_matrix[Gerudo, Akkala] = 99999999
distance_matrix[Gerudo, Central] = 99999999
distance_matrix[Gerudo, DuelingPeaks] = 99999999
distance_matrix[Gerudo, Eldin] = 99999999
distance_matrix[Gerudo, Faron] = 99999999
distance_matrix[Gerudo, Gerudo] = 99999999
distance_matrix[Gerudo, GreatPlateau] = 99999999
distance_matrix[Gerudo, Hateno] = 99999999
distance_matrix[Gerudo, Hebra] = 99999999
distance_matrix[Gerudo, Lake] = 99999999
distance_matrix[Gerudo, Lanayru] = 99999999
distance_matrix[Gerudo, Ridgeland] = 99999999
distance_matrix[Gerudo, Tabantha] = 99999999
distance_matrix[Gerudo, Wasteland] = 99999999
distance_matrix[Gerudo, Woodland] = 99999999

distance_matrix[Hateno, DuelingPeaks] = 99999999
distance_matrix[Hateno, Faron] = 99999999
distance_matrix[Hateno, Lake] = 99999999
distance_matrix[Hateno, Lanayru] = 99999999
distance_matrix[Hateno, Woodland] = 99999999  # maybe

distance_matrix[Lake, Hateno] = 99999999

distance_matrix[Lanayru, Faron] = 99999999
distance_matrix[Lanayru, Hateno] = 99999999

distance_matrix[Tabantha, Gerudo] = 99999999

distance_matrix[Woodland, Hateno] = 99999999  # maybe

# 100 m/s
# 3 m/s

min_distance, shortest_path = held_karp_tsp(distance_matrix)

print(calls)

print(f"Min Distance using Held-Karp: {min_distance}")
print("Shortest Path:", [towers[i].name for i in shortest_path])

plt.figure(figsize=(10, 8))

# Plot the background image
background_image_path = "base.png"
background_img = plt.imread(background_image_path)

# Plot the background image
plt.imshow(
    background_img,
    extent=[
        -6000,
        6000,
        -5000,
        5000,
    ],
)

# Plot the towers
x_values, y_values = tower_positions[:, 0], tower_positions[:, 1]
plt.scatter(x_values, -y_values, c="red", s=50, label="Towers")

# Plot the path
path_x = [towers[i].x for i in shortest_path]
path_y = [-towers[i].y for i in shortest_path]
plt.plot(
    path_x,
    path_y,
    marker="o",
    linestyle="-",
    color="blue",
    linewidth=2,
    markersize=8,
    label="Shortest Path",
)

# Customize the plot
plt.title("Traveling Salesman Problem - Shortest Path")
plt.legend()
plt.show()
