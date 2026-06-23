# VectorCore HNSW Search-Path Visualization

An interactive **React + D3** dashboard that animates how the C++ `HnswIndex`
answers a query: the greedy descent through the sparse upper layers, the
`efSearch` beam search at layer 0, and the final top-k selection — over a 2D
(PCA) projection of the real graph. A side panel shows live latency, recall@k,
and how few of the N nodes the search actually visits.

## How it works
1. **`export_graph.py`** builds a small `HnswIndex`, projects the vectors to 2D
   (PCA), and traces a single query through the *real* graph (read via the
   `entry_point` / `max_level` / `node_level` / `neighbors` accessors exposed by
   the C++ engine), recording the visit order and the parent→child traversal
   tree. It writes `public/graph_data.json`.
2. **The React app** (`src/App.jsx`) loads that JSON and animates it with D3
   scales + zoom, React-rendered SVG, and playback controls.

## Run it
```bash
# 1) Generate the data (needs vectorcore installed: pip install . at repo root)
python viz/export_graph.py

# 2) Install + run the front-end
cd viz
npm install
npm run dev        # open the printed localhost URL
# or: npm run build && npm run preview
```

## Controls
- **Play / Pause / Restart**, single **Step**, **Speed** and **Timeline** sliders.
- Drag to pan, scroll to zoom the graph.
