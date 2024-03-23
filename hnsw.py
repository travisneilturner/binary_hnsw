from __future__ import annotations
from pydantic import BaseModel
from typing import Dict, Set, List

import random
import numpy as np

def hamming_distance_numpy(a: np.ndarray, b: np.ndarray) -> int:
    print(f"a.shape: {a.shape}, b.shape: {b.shape}")
    assert a.shape == b.shape, "Arrays must be of the same shape"
    # Perform bitwise XOR between a and b, then count the set bits in the result.
    xor_result = np.bitwise_xor(a, b)
    count = np.sum(np.unpackbits(xor_result.astype(np.uint8)))
    return count

class HNSWNode(BaseModel):
    neighbors: Dict[int, Set[HNSWNode]]
    point: List[int]
    id: int

    def __eq__(self, other):
        return self.point == other.point

    def __hash__(self):
        return hash(tuple(self.point))

class HNSW(BaseModel):
    max_layers: int
    layers: Dict[int, Set[HNSWNode]] = {}
    num_nodes: int = 0

    def _insert(self, point:List[int]):
        node_id = self.num_nodes + 1
        self.num_nodes += 1

        node = HNSWNode(neighbors={}, point=point, id=node_id)

        chosen_layer = self._decide_layer_for_new_node()

        # Add node to each layer
        for layer in range(chosen_layer, -1, -1):
            if layer not in node.neighbors:
                node.neighbors[layer] = set()
            if layer not in self.layers:
                self.layers[layer] = set()
            self.layers[layer].add(node)
            for nearest in self._find_nearest_k_in_layer(node, layer, 10):
                if layer not in nearest.neighbors:
                    nearest.neighbors[layer] = set()
                nearest.neighbors[layer].add(node)
                node.neighbors[layer].add(nearest)

    def _decide_layer_for_new_node(self):
        r = random.random()  # Generate a random number between 0 and 1
        layer = 0
        while r < 0.5 and layer < self.max_layers:
            layer += 1
            r = random.random()
        return layer

    def _find_nearest_k_in_layer(self, origin: HNSWNode, layer: int, k: int = 1) -> List[HNSWNode]:
       # Store (distance, neighbor) tuples in a list
        distance_neighbor_pairs = []

        # Add origin's neighbors in the layer to the list
        for neighbor in origin.neighbors.get(layer, []):
            distance = hamming_distance_numpy(np.array(origin.point), np.array(neighbor.point))
            distance_neighbor_pairs.append((distance, neighbor))

        all_nodes_in_layer = set(self.layers[layer])
        remaining_neighbors = all_nodes_in_layer - set(origin.neighbors.get(layer, []))
        
        # Add remaining neighbors to the list
        for remainingNeighbor in remaining_neighbors:
            if origin.__eq__(remainingNeighbor):
                continue
            distance = hamming_distance_numpy(np.array(origin.point), np.array(remainingNeighbor.point))
            distance_neighbor_pairs.append((distance, remainingNeighbor))

        # Sort the combined list by distance
        distance_neighbor_pairs.sort(key=lambda x: x[0])

        # Take the first k elements after sorting
        nearest_neighbors = [neighbor for _, neighbor in distance_neighbor_pairs[:k]]

        return nearest_neighbors

    def search(self, point: List[int]) -> List[int]:
        nearest_neighbor_for_layer = (None, float("inf"))
        for layer in range(self.max_layers, -1, -1):
            if nearest_neighbor_for_layer[0] is None:
                searchNodes = self.layers.get(layer, []) 
            else:
                searchNodes = list(nearest_neighbor_for_layer[0].neighbors.get(layer, []))
             # Add origin's neighbors in the layer to the list
            for node in searchNodes:
                distance = hamming_distance_numpy(np.array(node.point), np.array(point))
                if distance < nearest_neighbor_for_layer[1]:
                    nearest_neighbor_for_layer = (node, distance)
        return nearest_neighbor_for_layer[0].point

def main():
    data = np.random.uniform(low=-1, high=1, size=(100, 128))

    hnsw = HNSW(max_layers=10) 
    bemb = np.where(data < 0, 0, 1).astype(np.uint8)
    bemb = np.packbits(bemb).reshape(bemb.shape[0], -1)

    # insert each row in bemb
    for row in bemb:
        hnsw._insert(row.tolist())
    
    query = np.random.uniform(low=-1, high=1, size=(1, 128))
    bemb = np.where(query < 0, 0, 1).astype(np.uint8)
    bemb = np.packbits(bemb).reshape(bemb.shape[0], -1)
    found = hnsw.search(bemb.reshape(-1).tolist())

    print(found)

if __name__ == "__main__":
    main()

