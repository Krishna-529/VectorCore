#include <cassert>
#include <cstdint>
#include <vector>

#include "vectorcore/bruteforce_index.h"

int main() {
  constexpr std::size_t dim = 4;

  vectorcore::BruteForceIndex index(dim, vectorcore::Metric::L2_SQUARED);

  // Two easy vectors.
  const float data[] = {
      0.f, 0.f, 0.f, 0.f,
      1.f, 0.f, 0.f, 0.f,
      0.f, 1.f, 0.f, 0.f,
  };

  index.add(data, 3);
  assert(index.size() == 3);

  const float q[] = {1.f, 0.f, 0.f, 0.f};
  std::uint64_t ids[2];
  float scores[2];

  index.search(q, 2, ids, scores);

  // Nearest neighbor should be the [1,0,0,0] vector (id 1).
  assert(ids[0] == 1);
  return 0;
}
