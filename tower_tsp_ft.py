import numpy as np
from collections import namedtuple
from enum import Enum
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
from itertools import combinations
from time import time


class TowerIdx(Enum):
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


Tower = namedtuple("Tower", "x z y")

towers: list[Tower] = [
    Tower(3308, 1500, 524),  # Akkala
    Tower(-788, -441, 128),  # Central
    Tower(1017, -1714, 114),  # DuelingPeaks
    Tower(2174, 1557, 439),  # Eldin
    Tower(1331, -3274, 200),  # Faron
    Tower(-3666, -1829, 401),  # Gerudo
    Tower(-560, -1695, 177),  # GreatPlateau
    Tower(2736, -2135, 266),  # Hateno
    Tower(-2173, 2034, 459),  # Hebra
    Tower(-32, -2963, 211),  # Lake
    Tower(2258, 109, 241),  # Lanayru
    Tower(-1755, 775, 259),  # Ridgeland
    Tower(-3614, 990, 375),  # Tabantha
    Tower(-2306, -2439, 460),  # Wasteland
    Tower(884, 1607, 280),  # Woodland
]


def calculate_time(tower_start: Tower, tower_end: Tower) -> float:
    BLSS_SPEED = 100
    CLIMB_SPEED = 3
    FALL_SPEED = 10

    distance = (
        np.hypot(tower_end.x - tower_start.x, tower_end.z - tower_start.z) / BLSS_SPEED
    )
    height_diff = tower_end.y - tower_start.y
    climb = max(0, height_diff) / CLIMB_SPEED
    fall = abs(min(0, height_diff)) / FALL_SPEED

    return distance + climb + fall


distance_matrix = np.zeros((len(towers), len(towers)))
for i in range(len(towers)):
    for j in range(len(towers)):
        distance_matrix[i, j] = calculate_time(towers[i], towers[j])


def remove_route(tower1: TowerIdx, tower2: TowerIdx):
    distance_matrix[tower1.value, tower2.value] = np.nan


remove_route(TowerIdx.Akkala, TowerIdx.Akkala)
remove_route(TowerIdx.Akkala, TowerIdx.Gerudo)

remove_route(TowerIdx.Central, TowerIdx.Akkala)
remove_route(TowerIdx.Central, TowerIdx.Central)
remove_route(TowerIdx.Central, TowerIdx.Eldin)
remove_route(TowerIdx.Central, TowerIdx.Faron)
remove_route(TowerIdx.Central, TowerIdx.Gerudo)
remove_route(TowerIdx.Central, TowerIdx.Hateno)
remove_route(TowerIdx.Central, TowerIdx.Hebra)
# remove_route(TowerIdx.Central, TowerIdx.Lake) # base/fast
remove_route(TowerIdx.Central, TowerIdx.Lanayru)
remove_route(TowerIdx.Central, TowerIdx.Ridgeland)
remove_route(TowerIdx.Central, TowerIdx.Tabantha)
remove_route(TowerIdx.Central, TowerIdx.Wasteland)
remove_route(TowerIdx.Central, TowerIdx.Woodland)

remove_route(TowerIdx.DuelingPeaks, TowerIdx.Akkala)
remove_route(TowerIdx.DuelingPeaks, TowerIdx.DuelingPeaks)
remove_route(TowerIdx.DuelingPeaks, TowerIdx.Eldin)
remove_route(TowerIdx.DuelingPeaks, TowerIdx.Faron)
remove_route(TowerIdx.DuelingPeaks, TowerIdx.Gerudo)
remove_route(TowerIdx.DuelingPeaks, TowerIdx.Hateno)
remove_route(TowerIdx.DuelingPeaks, TowerIdx.Hebra)
remove_route(TowerIdx.DuelingPeaks, TowerIdx.Lake)
remove_route(TowerIdx.DuelingPeaks, TowerIdx.Lanayru)
remove_route(TowerIdx.DuelingPeaks, TowerIdx.Ridgeland)
remove_route(TowerIdx.DuelingPeaks, TowerIdx.Tabantha)
remove_route(TowerIdx.DuelingPeaks, TowerIdx.Wasteland)
remove_route(TowerIdx.DuelingPeaks, TowerIdx.Woodland)

# remove_route(TowerIdx.Eldin, TowerIdx.Akkala) # base/slow - but included since common
remove_route(TowerIdx.Eldin, TowerIdx.Eldin)
remove_route(TowerIdx.Eldin, TowerIdx.Gerudo)

remove_route(TowerIdx.Faron, TowerIdx.Akkala)
# remove_route(TowerIdx.Faron, TowerIdx.DuelingPeaks)  # curved/hard
remove_route(TowerIdx.Faron, TowerIdx.Eldin)
remove_route(TowerIdx.Faron, TowerIdx.Faron)
remove_route(TowerIdx.Faron, TowerIdx.Gerudo)
remove_route(TowerIdx.Faron, TowerIdx.Hateno)
remove_route(TowerIdx.Faron, TowerIdx.Hebra)
remove_route(TowerIdx.Faron, TowerIdx.Lanayru)
remove_route(TowerIdx.Faron, TowerIdx.Tabantha)
remove_route(TowerIdx.Faron, TowerIdx.Wasteland)
remove_route(TowerIdx.Faron, TowerIdx.Woodland)  # curved/hard - not proven

remove_route(TowerIdx.Gerudo, TowerIdx.Akkala)
remove_route(TowerIdx.Gerudo, TowerIdx.Central)
remove_route(TowerIdx.Gerudo, TowerIdx.DuelingPeaks)
remove_route(TowerIdx.Gerudo, TowerIdx.Eldin)
remove_route(TowerIdx.Gerudo, TowerIdx.Faron)
remove_route(TowerIdx.Gerudo, TowerIdx.Gerudo)
remove_route(TowerIdx.Gerudo, TowerIdx.GreatPlateau)
remove_route(TowerIdx.Gerudo, TowerIdx.Hateno)
remove_route(TowerIdx.Gerudo, TowerIdx.Hebra)
remove_route(TowerIdx.Gerudo, TowerIdx.Lake)
remove_route(TowerIdx.Gerudo, TowerIdx.Lanayru)
remove_route(TowerIdx.Gerudo, TowerIdx.Ridgeland)
remove_route(TowerIdx.Gerudo, TowerIdx.Tabantha)
remove_route(TowerIdx.Gerudo, TowerIdx.Wasteland)
remove_route(TowerIdx.Gerudo, TowerIdx.Woodland)

remove_route(TowerIdx.GreatPlateau, TowerIdx.Akkala)
remove_route(TowerIdx.GreatPlateau, TowerIdx.Eldin)
remove_route(TowerIdx.GreatPlateau, TowerIdx.Gerudo)
remove_route(TowerIdx.GreatPlateau, TowerIdx.GreatPlateau)
remove_route(TowerIdx.GreatPlateau, TowerIdx.Hateno)
remove_route(TowerIdx.GreatPlateau, TowerIdx.Hebra)
remove_route(TowerIdx.GreatPlateau, TowerIdx.Lanayru)
# remove_route(TowerIdx.GreatPlateau, TowerIdx.Ridgeland)  # base/slow
remove_route(TowerIdx.GreatPlateau, TowerIdx.Tabantha)
remove_route(TowerIdx.GreatPlateau, TowerIdx.Wasteland)
# remove_route(TowerIdx.GreatPlateau, TowerIdx.Woodland)  # base/slow

remove_route(TowerIdx.Hateno, TowerIdx.Akkala)
remove_route(TowerIdx.Hateno, TowerIdx.Central)
remove_route(TowerIdx.Hateno, TowerIdx.Eldin)
remove_route(TowerIdx.Hateno, TowerIdx.Faron)
remove_route(TowerIdx.Hateno, TowerIdx.Gerudo)
remove_route(TowerIdx.Hateno, TowerIdx.Hateno)
remove_route(TowerIdx.Hateno, TowerIdx.Hebra)
remove_route(TowerIdx.Hateno, TowerIdx.Tabantha)
remove_route(TowerIdx.Hateno, TowerIdx.Wasteland)
# remove_route(TowerIdx.Hateno, TowerIdx.Woodland)  # curved/hard

# remove_route(TowerIdx.Hebra, TowerIdx.Akkala)  # base/slow
remove_route(TowerIdx.Hebra, TowerIdx.Gerudo)
remove_route(TowerIdx.Hebra, TowerIdx.Hebra)

remove_route(TowerIdx.Lake, TowerIdx.Akkala)
remove_route(TowerIdx.Lake, TowerIdx.Eldin)
remove_route(TowerIdx.Lake, TowerIdx.Gerudo)
remove_route(TowerIdx.Lake, TowerIdx.Hateno)
remove_route(TowerIdx.Lake, TowerIdx.Hebra)
remove_route(TowerIdx.Lake, TowerIdx.Lake)
remove_route(TowerIdx.Lake, TowerIdx.Tabantha)
remove_route(TowerIdx.Lake, TowerIdx.Wasteland)

remove_route(TowerIdx.Lanayru, TowerIdx.Akkala)
remove_route(TowerIdx.Lanayru, TowerIdx.DuelingPeaks)
remove_route(TowerIdx.Lanayru, TowerIdx.Eldin)
remove_route(TowerIdx.Lanayru, TowerIdx.Faron)
remove_route(TowerIdx.Lanayru, TowerIdx.Gerudo)
# remove_route(TowerIdx.Lanayru, TowerIdx.Hateno)  # curved/hard
remove_route(TowerIdx.Lanayru, TowerIdx.Hebra)
remove_route(TowerIdx.Lanayru, TowerIdx.Lanayru)
remove_route(TowerIdx.Lanayru, TowerIdx.Tabantha)  # possible?
remove_route(TowerIdx.Lanayru, TowerIdx.Wasteland)

remove_route(TowerIdx.Ridgeland, TowerIdx.Akkala)
remove_route(TowerIdx.Ridgeland, TowerIdx.Eldin)
remove_route(TowerIdx.Ridgeland, TowerIdx.Gerudo)
remove_route(TowerIdx.Ridgeland, TowerIdx.Hebra)
remove_route(TowerIdx.Ridgeland, TowerIdx.Ridgeland)
remove_route(TowerIdx.Ridgeland, TowerIdx.Tabantha)
remove_route(TowerIdx.Ridgeland, TowerIdx.Wasteland)

remove_route(TowerIdx.Tabantha, TowerIdx.Akkala)
# remove_route(TowerIdx.Tabantha, TowerIdx.Eldin)  # base/fast
remove_route(TowerIdx.Tabantha, TowerIdx.Gerudo)
# remove_route(TowerIdx.Tabantha, TowerIdx.Hebra)  # base/slow
remove_route(TowerIdx.Tabantha, TowerIdx.Lake)
remove_route(TowerIdx.Tabantha, TowerIdx.Tabantha)
remove_route(TowerIdx.Tabantha, TowerIdx.Wasteland)  # curved/hard base/slow

# remove_route(TowerIdx.Wasteland, TowerIdx.Akkala) # base/slow
remove_route(TowerIdx.Wasteland, TowerIdx.Tabantha)  # curve/hard - unproven
remove_route(TowerIdx.Wasteland, TowerIdx.Wasteland)

remove_route(TowerIdx.Woodland, TowerIdx.Akkala)
remove_route(TowerIdx.Woodland, TowerIdx.Central)
remove_route(TowerIdx.Woodland, TowerIdx.Eldin)
remove_route(TowerIdx.Woodland, TowerIdx.Faron)
remove_route(TowerIdx.Woodland, TowerIdx.Gerudo)
remove_route(TowerIdx.Woodland, TowerIdx.Hebra)
remove_route(TowerIdx.Woodland, TowerIdx.Ridgeland)
# remove_route(TowerIdx.Woodland, TowerIdx.Tabantha)  # possible from top of the skull
distance_matrix[TowerIdx.Woodland.value, TowerIdx.Tabantha.value] += 10
remove_route(TowerIdx.Woodland, TowerIdx.Wasteland)
remove_route(TowerIdx.Woodland, TowerIdx.Woodland)


def print_tower_distance_matrix(distance_matrix: np.ndarray):
    separator_line = "-" * (distance_matrix.shape[0] + 1) * 6

    print(
        "\n     |"
        + "|".join(
            f"{TowerIdx(i).name[:5]:>5}" for i in range(distance_matrix.shape[0])
        )
        + "\n"
        + separator_line
    )

    for i, row in enumerate(distance_matrix):
        print(f"{TowerIdx(i).name[:5]:>5}|", end="")
        print(
            "|".join(f"{int(value) if not np.isnan(value) else '':>5}" for value in row)
        )
        print(separator_line)


def matrix_to_graph(matrix: np.ndarray) -> nx.DiGraph:
    G = nx.DiGraph()
    for i, row in enumerate(matrix):
        for j, weight in enumerate(row):
            if not np.isnan(weight):
                G.add_edge(i, j, weight=weight)
    print(G)
    return G


def find_minimum_spanning_arborescence(
    matrix: np.ndarray,
    n: int = 15,
    required_nodes: set[int] = {TowerIdx.GreatPlateau.value},
) -> nx.DiGraph:
    # sums = np.nansum(matrix, axis=0)
    # print_tower_distance_matrix(matrix - (sums - matrix))
    updated_weights = matrix - np.nanmin(matrix, axis=0)
    print_tower_distance_matrix(updated_weights)
    minidxs = np.nanargmin(updated_weights, axis=0)
    print(minidxs)
    return [
        (minidx, i, {"weight": matrix[minidx, i]}) for i, minidx in enumerate(minidxs)
    ]


# def find_minimum_spanning_arborescence(
#     matrix: np.ndarray,
#     n: int = 15,
#     required_nodes: set[int] = {TowerIdx.GreatPlateau.value},
# ) -> nx.DiGraph:
#     TEMP_ROOT_NODE = 99

#     G = matrix_to_graph(matrix)

#     # add edge for setting the correct root
#     G.add_edge(TEMP_ROOT_NODE, TowerIdx.GreatPlateau.value, weight=0)

#     best_arborescence = None
#     best_weight = float("inf")

#     if n == len(matrix):
#         best_arborescence = nx.DiGraph(
#             nx.minimum_spanning_arborescence(G, attr="weight").edges(data=True)
#         )
#     else:
#         for subset_nodes in combinations(range(len(matrix)), n):
#             # check that subgraph will have all the required nodes
#             if not required_nodes.issubset(subset_nodes):
#                 continue

#             edges = []
#             weight = float("inf")
#             try:
#                 edges = nx.minimum_spanning_arborescence(
#                     G.subgraph(subset_nodes + (TEMP_ROOT_NODE,)), attr="weight"
#                 ).edges(data=True)
#                 weight = sum(edge[2]["weight"] for edge in edges)
#             except nx.exception.NetworkXException:
#                 continue

#             if weight < best_weight:
#                 best_arborescence = nx.DiGraph(edges)
#                 best_weight = weight

#     best_arborescence.remove_edge(TEMP_ROOT_NODE, TowerIdx.GreatPlateau.value)
#     return best_arborescence


print_tower_distance_matrix(distance_matrix)

start = time()
min_graph = find_minimum_spanning_arborescence(
    distance_matrix,
    15,
    # {TowerIdx.GreatPlateau.value, TowerIdx.Akkala.value, TowerIdx.Gerudo.value},
)
print(f"\nSearch took: {round(time() - start, 2)} s")
# min_graph = min_graph.edges(data=True)

print("\nMinimum Spanning Arborescence Edges:")
for start_node, end_node, _ in min_graph:
    print(TowerIdx(start_node).name, "->", TowerIdx(end_node).name)

# teleport_cost = 10

plt.figure(figsize=(11, 9))

background_image_path = "base.png"
background_img = plt.imread(background_image_path)
plt.imshow(
    background_img,
    extent=[
        -6000,
        6000,
        -5000,
        5000,
    ],
)

tower_positions = np.array([(tower.x, tower.z) for tower in towers])

plt.scatter(tower_positions[:, 0], tower_positions[:, 1], c="red", s=50, label="Towers")

for i, tower in enumerate(towers):
    plt.text(
        tower.x,
        tower.z + 200,
        f"{TowerIdx(i).name}\ny:{tower.y}",
        fontsize=9,
        # backgroundcolor="black",
        color="white",
        ha="center",
        va="bottom",
    )


total_distance = 0
total_time = 0
for start_node, end_node, data in min_graph:
    tower_start = towers[start_node]
    tower_end = towers[end_node]
    dx = tower_end.x - tower_start.x
    dz = tower_end.z - tower_start.z

    total_time += data["weight"]
    total_distance += np.hypot(dx, dz)

    plt.arrow(
        tower_start.x,
        tower_start.z,
        dx,
        dz,
        antialiased=True,
        head_width=200,
        head_length=200,
        length_includes_head=True,
    )

print(f"\nTotal Distance: {total_distance / 1000:.2f} km")
print(f"Total Time: {int(total_time / 60)} min {int(total_time % 60)} s")

plt.tight_layout()
plt.show()
