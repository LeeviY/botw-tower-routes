#include "matplot/matplot.h"
#include <cmath>
#include <iostream>
#include <tuple>
#include <vector>

namespace plt = matplot;

enum TowerIdx {
    Akkala = 0,
    Central = 1,
    DuelingPeaks = 2,
    Eldin = 3,
    Faron = 4,
    Gerudo = 5,
    GreatPlateau = 6,
    Hateno = 7,
    Hebra = 8,
    Lake = 9,
    Lanayru = 10,
    Ridgeland = 11,
    Tabantha = 12,
    Wasteland = 13,
    Woodland = 14,
};

struct Tower {
    double x, y, h;
};

std::vector<Tower> towers = {
    Tower{3308, -1500, 524},    // Akkala
    Tower{-788, 441, 128},      // Central
    Tower{1017, 1714, 114},     // DuelingPeaks
    Tower{2174, -1557, 439},    // Eldin
    Tower{1331, 3274, 200},     // Faron
    Tower{-3666, 1829, 401},    // Gerudo
    Tower{-560, 1695, 177},     // GreatPlateau
    Tower{2736, 2135, 266},     // Hateno
    Tower{-2173, -2034, 459},   // Hebra
    Tower{-32, 2963, 211},      // Lake
    Tower{2258, -109, 241},     // Lanayru
    Tower{-1755, -775, 259},    // Ridgeland
    Tower{-3614, -990, 375},    // Tabantha
    Tower{-2306, 2439, 460},    // Wasteland
    Tower{884, -1607, 280},     // Woodland
};

static std::vector<std::vector<double>>
calcualteTowerDistanceMatrix(std::vector<Tower> towers) {
    std::vector<std::vector<double>> distanceMatrix(
        towers.size(), std::vector<double>(towers.size()));

    for (size_t i = 0; i < towers.size(); ++i) {
        for (size_t j = 0; j < towers.size(); ++j) {
            double distance = std::sqrt(std::pow(towers[i].x - towers[j].x, 2) +
                                        std::pow(towers[i].y - towers[j].y, 2));
            double heightDiff = std::max<double>(0, towers[i].h - towers[j].h);
            distanceMatrix[i][j] = distance / 100 + heightDiff / 3;
        }
    }

    distanceMatrix[TowerIdx::DuelingPeaks][TowerIdx::Faron] = 99999999;
    distanceMatrix[TowerIdx::DuelingPeaks][TowerIdx::Hateno] = 99999999;
    distanceMatrix[TowerIdx::Faron][TowerIdx::DuelingPeaks] = 99999999;
    distanceMatrix[TowerIdx::Faron][TowerIdx::Hateno] = 99999999;
    distanceMatrix[TowerIdx::Faron][TowerIdx::Lanayru] = 99999999;
    distanceMatrix[TowerIdx::Gerudo][TowerIdx::Akkala] = 99999999;
    distanceMatrix[TowerIdx::Gerudo][TowerIdx::Central] = 99999999;
    distanceMatrix[TowerIdx::Gerudo][TowerIdx::DuelingPeaks] = 99999999;
    distanceMatrix[TowerIdx::Gerudo][TowerIdx::Eldin] = 99999999;
    distanceMatrix[TowerIdx::Gerudo][TowerIdx::Faron] = 99999999;
    distanceMatrix[TowerIdx::Gerudo][TowerIdx::Gerudo] = 99999999;
    distanceMatrix[TowerIdx::Gerudo][TowerIdx::GreatPlateau] = 99999999;
    distanceMatrix[TowerIdx::Gerudo][TowerIdx::Hateno] = 99999999;
    distanceMatrix[TowerIdx::Gerudo][TowerIdx::Hebra] = 99999999;
    distanceMatrix[TowerIdx::Gerudo][TowerIdx::Lake] = 99999999;
    distanceMatrix[TowerIdx::Gerudo][TowerIdx::Lanayru] = 99999999;
    distanceMatrix[TowerIdx::Gerudo][TowerIdx::Ridgeland] = 99999999;
    distanceMatrix[TowerIdx::Gerudo][TowerIdx::Tabantha] = 99999999;
    distanceMatrix[TowerIdx::Gerudo][TowerIdx::Wasteland] = 99999999;
    distanceMatrix[TowerIdx::Gerudo][TowerIdx::Woodland] = 99999999;
    distanceMatrix[TowerIdx::Hateno][TowerIdx::DuelingPeaks] = 99999999;
    distanceMatrix[TowerIdx::Hateno][TowerIdx::Faron] = 99999999;
    distanceMatrix[TowerIdx::Hateno][TowerIdx::Lake] = 99999999;
    distanceMatrix[TowerIdx::Hateno][TowerIdx::Lanayru] = 99999999;
    distanceMatrix[TowerIdx::Hateno][TowerIdx::Woodland] = 99999999;   // maybe
    distanceMatrix[TowerIdx::Lake][TowerIdx::Hateno] = 99999999;
    distanceMatrix[TowerIdx::Lanayru][TowerIdx::Faron] = 99999999;
    distanceMatrix[TowerIdx::Lanayru][TowerIdx::Hateno] = 99999999;
    distanceMatrix[TowerIdx::Tabantha][TowerIdx::Gerudo] = 99999999;
    distanceMatrix[TowerIdx::Woodland][TowerIdx::Hateno] = 99999999;   // maybe

    return distanceMatrix;
}

// Function to find the minimum key value node from the set of vertices not yet
// included in MST
int minKey(const std::vector<int> &key, const std::vector<bool> &mstSet,
           int vertices) {
    int min = INT_MAX, min_index;

    for (int v = 0; v < vertices; ++v) {
        if (!mstSet[v] && key[v] < min) {
            min = key[v];
            min_index = v;
        }
    }

    return min_index;
}

// Function to print the minimum spanning tree
void printMST(const std::vector<int> &parent,
              const std::vector<std::vector<double>> &graph, int vertices) {
    std::cout << "Edge \tWeight\n";
    for (int i = 1; i < vertices; ++i) {
        std::cout << parent[i] << " - " << i << "\t" << graph[i][parent[i]]
                  << "\n";
    }
}

// Function to perform Prim's algorithm to find the minimum spanning tree
void primMST(const std::vector<std::vector<double>> &graph, int vertices) {
    // Vector to store the parent of each node in the MST
    std::vector<int> parent(vertices, -1);

    // Vector to store the key values used to pick the minimum weight edge
    std::vector<int> key(vertices, INT_MAX);

    // Vector to track whether a vertex is included in MST
    std::vector<bool> mstSet(vertices, false);

    // Start with the first vertex
    key[0] = 0;

    for (int count = 0; count < vertices - 1; ++count) {
        int u = minKey(key, mstSet, vertices);
        mstSet[u] = true;

        for (int v = 0; v < vertices; ++v) {
            if (graph[u][v] && !mstSet[v] && graph[u][v] < key[v]) {
                parent[v] = u;
                key[v] = graph[u][v];
            }
        }
    }

    printMST(parent, graph, vertices);

    // Plot the minimum spanning tree
    for (int i = 1; i < vertices; ++i) {
        plt::plot({(double) parent[i], (double) i},
                  {(double) parent[i], (double) i}, "bo-");
    }

    // Display the plot
    plt::show();
}

int main() {
    auto distanceMatrix = calcualteTowerDistanceMatrix(towers);

    primMST(distanceMatrix, towers.size());

    return 0;
}
