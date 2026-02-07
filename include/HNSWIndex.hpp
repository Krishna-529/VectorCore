#pragma once

#include <vector>

namespace vectorcore {

/*
Algorithm Description
---------------------
HNSW (Hierarchical Navigable Small World) is a multi-layer proximity graph.

Key ideas you can explain in an interview:

- Layer 0 contains all nodes.
  This is the dense base graph where final nearest-neighbor refinement happens.

- Upper layers act as an express lane (Skip List analogy).
  Each higher layer contains fewer nodes. Searching these sparse layers quickly
  moves you close to the target region, similar to how a skip list "skips" over
  many elements at once.

- Greedy Search: Move to the neighbor closest to the target.
  Starting from an entry point, repeatedly jump to the neighbor that improves
  distance to the query until no neighbor is better, then descend to the next
  layer and repeat.

This file is intentionally architecture-focused (WIP implementation): we provide
just enough structure to compile and to demonstrate understanding of the design.
*/

struct HnswNode {
  int id = 0;

  // connections[layer] = list of neighbor node indices (or IDs) at that layer.
  //
  // Note: This is a vector-of-vectors, but it's graph metadata, not embedding
  // storage. The flat-memory constraint applies to embedding vectors.
  std::vector<std::vector<int>> connections;
};

class HNSWIndex {
public:
  HNSWIndex() = default;

  // Dummy insert: appends the node to an internal list.
  // A real HNSW insert would:
  // - choose a random max level for the new node
  // - navigate from top layer down to find entry points
  // - connect the node to up to M neighbors per layer
  void insert(const HnswNode& node) { nodes_.push_back(node); }

  const std::vector<HnswNode>& nodes() const noexcept { return nodes_; }

private:
  std::vector<HnswNode> nodes_;
};

} // namespace vectorcore
