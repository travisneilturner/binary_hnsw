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
    neighbors: Dict[int, Set[int]]
    point: List[int]
    id: int

    def __eq__(self, other):
        return self.point == other.point

    def __hash__(self):
        return hash(tuple(self.point))

class HNSW(BaseModel):
    max_layers: int
    layers: Dict[int, Set[int]] = {}
    nodes: Dict[int, HNSWNode] = {}
    num_nodes: int = 0

    def _insert(self, point:List[int]):
        node_id = self.num_nodes
        self.num_nodes += 1

        node = HNSWNode(neighbors={}, point=point, id=node_id)

        chosen_layer = self._decide_layer_for_new_node()

        # Add node to each layer
        for layer in range(chosen_layer, -1, -1):
            if layer not in node.neighbors:
                node.neighbors[layer] = set()
            if layer not in self.layers:
                self.layers[layer] = set()
            self.layers[layer].add(node.id)
            self.nodes[node.id] = node
            for nearest in self._find_nearest_k_in_layer(node, layer, 10):
                if layer not in nearest.neighbors:
                    nearest.neighbors[layer] = set()
                nearest.neighbors[layer].add(node.id)
                node.neighbors[layer].add(nearest.id)

    def _decide_layer_for_new_node(self):
        r = random.random()  # Generate a random number between 0 and 1
        layer = 0
        while r < 0.5 and layer < self.max_layers:
            layer += 1
            r = random.random()
        return layer

    def _find_nearest_k_in_layer(self, origin: HNSWNode, layer: int, k: int = 1) -> List[HNSWNode]:
       # Store (distance, neighbor) tuples in a list
        distance_neighbor_id_pairs = []

        all_node_ids_in_layer = set(self.layers[layer])
        remaining_neighbor_ids = all_node_ids_in_layer - set(origin.neighbors.get(layer, []))
        
        # Add remaining neighbors to the list
        for remaining_neighbor_id in remaining_neighbor_ids:
            if self.nodes[origin.id].__eq__(self.nodes[remaining_neighbor_id]):
                continue
            distance = hamming_distance_numpy(np.array(origin.point), np.array(self.nodes[remaining_neighbor_id].point))
            distance_neighbor_id_pairs.append((distance, remaining_neighbor_id))

        # Sort the combined list by distance
        distance_neighbor_id_pairs.sort(key=lambda x: x[0])

        # Take the first k elements after sorting
        nearest_neighbors = [self.nodes[neighbor_id] for _, neighbor_id in distance_neighbor_id_pairs[:k]]

        return nearest_neighbors

    def search(self, point: List[int]) -> List[int]:
        nearest_neighbor_for_layer = (None, float("inf"))
        for layer in range(self.max_layers, -1, -1):
            if nearest_neighbor_for_layer[0] is None:
                search_ids = self.layers.get(layer, []) 
            else:
                search_ids = list(self.nodes[nearest_neighbor_for_layer[0]].neighbors.get(layer, []))
             # Add origin's neighbors in the layer to the list
            for id in search_ids:
                distance = hamming_distance_numpy(np.array(self.nodes[id].point), np.array(point))
                if distance < nearest_neighbor_for_layer[1]:
                    nearest_neighbor_for_layer = (id, distance)
        return self.nodes[nearest_neighbor_for_layer[0]].point

def main():
    data = np.random.uniform(low=-1, high=1, size=(100, 128))

    hnsw = HNSW(max_layers=10) 
    binary_vecs = np.where(data < 0, 0, 1).astype(np.uint8)
    binary_vecs = np.packbits(binary_vecs).reshape(binary_vecs.shape[0], -1)

    # insert each row in bemb
    for row in binary_vecs:
        hnsw._insert(row.tolist())
    
    query = np.random.uniform(low=-1, high=1, size=(1, 128))
    binary_vec = np.where(query < 0, 0, 1).astype(np.uint8)
    binary_vec = np.packbits(binary_vec).reshape(binary_vec.shape[0], -1)
    found = hnsw.search(binary_vec.reshape(-1).tolist())

    print(found)

if __name__ == "__main__":
    main()

