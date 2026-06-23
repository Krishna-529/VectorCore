import React, { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";

// Build a flat timeline of animation steps from the search trace:
//   descent steps (zoom-in through upper layers) -> layer-0 expansions -> done
function buildTimeline(trace) {
  const steps = [];
  trace.descent.forEach((d, i) => {
    steps.push({ kind: "descent", node: d.node, layer: d.layer, order: i });
  });
  trace.visited.forEach((node, i) => {
    steps.push({ kind: "visit", node, idx: i });
  });
  steps.push({ kind: "done" });
  return steps;
}

const clusterColor = d3.scaleOrdinal(d3.schemeCategory10);

export default function App() {
  const [data, setData] = useState(null);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(6); // steps per second
  const svgRef = useRef(null);
  const gRef = useRef(null);

  useEffect(() => {
    fetch("graph_data.json")
      .then((r) => r.json())
      .then(setData)
      .catch(() => setData("error"));
  }, []);

  const timeline = useMemo(
    () => (data && data !== "error" ? buildTimeline(data.trace) : []),
    [data]
  );

  // Geometry: map data coords into the viewport with padding.
  const W = 900, H = 720, PAD = 48;
  const scales = useMemo(() => {
    if (!data || data === "error") return null;
    const xs = data.nodes.map((n) => n.x).concat(data.query.x);
    const ys = data.nodes.map((n) => n.y).concat(data.query.y);
    const x = d3.scaleLinear().domain(d3.extent(xs)).range([PAD, W - PAD]);
    const y = d3.scaleLinear().domain(d3.extent(ys)).range([H - PAD, PAD]);
    return { x, y };
  }, [data]);

  // Playback loop.
  useEffect(() => {
    if (!playing) return;
    if (step >= timeline.length - 1) { setPlaying(false); return; }
    const id = setTimeout(() => setStep((s) => Math.min(s + 1, timeline.length - 1)),
                          1000 / speed);
    return () => clearTimeout(id);
  }, [playing, step, speed, timeline.length]);

  // Pan/zoom via d3-zoom on the <g> group.
  useEffect(() => {
    if (!svgRef.current || !gRef.current) return;
    const g = d3.select(gRef.current);
    const zoom = d3.zoom().scaleExtent([0.5, 6]).on("zoom", (e) => {
      g.attr("transform", e.transform);
    });
    d3.select(svgRef.current).call(zoom);
    return () => d3.select(svgRef.current).on(".zoom", null);
  }, [data]);

  if (data === "error") {
    return <div className="loading">Could not load graph_data.json — run <code>python viz/export_graph.py</code> first.</div>;
  }
  if (!data || !scales) {
    return <div className="loading">Loading search trace…</div>;
  }

  const { x, y } = scales;
  const { nodes, edges, query, trace, meta } = data;

  // Derive animation state from the current step.
  const visitedSet = new Set();
  const descentNodes = [];
  let currentNode = null;
  let phase = "Ready";
  for (let i = 0; i <= step && i < timeline.length; i++) {
    const s = timeline[i];
    if (s.kind === "descent") {
      descentNodes.push(s.node);
      visitedSet.add(s.node);
      currentNode = s.node;
      phase = s.layer > 0 ? `Greedy descent · layer ${s.layer}` : "Reached layer 0";
    } else if (s.kind === "visit") {
      visitedSet.add(s.node);
      currentNode = s.node;
      phase = `Beam search · layer 0 (ef=${meta.ef_search})`;
    } else if (s.kind === "done") {
      phase = "Done — top-k selected";
    }
  }
  const resultsSet = new Set(step >= timeline.length - 1 ? trace.results : []);

  // Descent path segments (entry point hopping down the layers).
  const descentSegs = [];
  for (let i = 1; i < descentNodes.length; i++) {
    descentSegs.push([descentNodes[i - 1], descentNodes[i]]);
  }

  // Traversal-tree edges discovered so far (parent -> child at layer 0).
  const liveEdges = trace.visited_edges.filter(
    ([a, b]) => visitedSet.has(a) && visitedSet.has(b)
  );

  const nodeById = (id) => nodes[id];
  const pct = timeline.length > 1 ? (step / (timeline.length - 1)) * 100 : 0;

  return (
    <div className="app">
      <div className="stage">
        <div className="title">
          <h1>HNSW Search Path</h1>
          <p>{meta.n.toLocaleString()} vectors · {meta.dim}-dim · M={meta.M} · {meta.metric.toUpperCase()}</p>
        </div>
        <div className="phase">{phase}</div>
        <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
          <g ref={gRef}>
            {/* Backdrop: layer-0 small-world graph (faint). */}
            <g stroke="#2a3040" strokeWidth={0.5} opacity={0.25}>
              {edges.map(([a, b], i) => (
                <line key={i} x1={x(nodeById(a).x)} y1={y(nodeById(a).y)}
                      x2={x(nodeById(b).x)} y2={y(nodeById(b).y)} />
              ))}
            </g>

            {/* Traversal tree discovered so far. */}
            <g stroke="#4f8cff" strokeWidth={1.4} opacity={0.8}>
              {liveEdges.map(([a, b], i) => (
                <line key={i} x1={x(nodeById(a).x)} y1={y(nodeById(a).y)}
                      x2={x(nodeById(b).x)} y2={y(nodeById(b).y)} />
              ))}
            </g>

            {/* Descent path (zoom-in across layers). */}
            <g stroke="#ffd166" strokeWidth={2.2} strokeDasharray="6 4" opacity={0.9}>
              {descentSegs.map(([a, b], i) => (
                <line key={i} x1={x(nodeById(a).x)} y1={y(nodeById(a).y)}
                      x2={x(nodeById(b).x)} y2={y(nodeById(b).y)} />
              ))}
            </g>

            {/* Nodes. */}
            <g>
              {nodes.map((n) => {
                const visited = visitedSet.has(n.id);
                const isResult = resultsSet.has(n.id);
                const isCurrent = n.id === currentNode && step < timeline.length - 1;
                const r = isResult ? 7 : isCurrent ? 7 : visited ? 5 : 2.6 + n.level * 1.1;
                let fill = visited ? "#cdd6e4" : clusterColor(n.cluster);
                let opacity = visited ? 1 : 0.5;
                if (isResult) fill = "#3ddc97";
                if (isCurrent) fill = "#ff5c5c";
                return (
                  <g key={n.id}>
                    {isResult && (
                      <circle cx={x(n.x)} cy={y(n.y)} r={11} fill="none"
                              stroke="#3ddc97" strokeWidth={1.5} opacity={0.7} />
                    )}
                    {isCurrent && (
                      <circle cx={x(n.x)} cy={y(n.y)} r={12} fill="none"
                              stroke="#ff5c5c" strokeWidth={1.5} opacity={0.6}>
                        <animate attributeName="r" values="8;14;8" dur="1s" repeatCount="indefinite" />
                      </circle>
                    )}
                    <circle cx={x(n.x)} cy={y(n.y)} r={r} fill={fill} opacity={opacity} />
                  </g>
                );
              })}
            </g>

            {/* Entry point marker. */}
            <circle cx={x(nodeById(trace.entry_point).x)} cy={y(nodeById(trace.entry_point).y)}
                    r={9} fill="none" stroke="#ffd166" strokeWidth={2} />

            {/* Query point (diamond). */}
            <g transform={`translate(${x(query.x)},${y(query.y)}) rotate(45)`}>
              <rect x={-7} y={-7} width={14} height={14} fill="#4f8cff" stroke="#fff" strokeWidth={1.5} />
            </g>
          </g>
        </svg>
      </div>

      <div className="side">
        <div className="card">
          <h2>Live Metrics</h2>
          <div className="metric"><span className="label">Latency</span>
            <span className="value accent">{meta.latency_ms.toFixed(3)} ms</span></div>
          <div className="metric"><span className="label">Recall@{meta.k}</span>
            <span className="value good">{(meta.recall * 100).toFixed(1)}%</span></div>
          <div className="metric"><span className="label">Nodes visited</span>
            <span className="value">{visitedSet.size} / {meta.n}</span></div>
          <div className="metric"><span className="label">Graph layers</span>
            <span className="value">{meta.max_level + 1}</span></div>
          <div className="metric"><span className="label">ef_search</span>
            <span className="value">{meta.ef_search}</span></div>
          <div className="progress"><div style={{ width: `${pct}%` }} /></div>
        </div>

        <div className="card">
          <h2>Playback</h2>
          <div className="controls">
            <div className="row">
              <button className="primary" onClick={() => {
                if (step >= timeline.length - 1) setStep(0);
                setPlaying((p) => !p);
              }}>{playing ? "Pause" : "Play"}</button>
              <button onClick={() => { setPlaying(false); setStep(0); }}>Restart</button>
            </div>
            <div className="row">
              <button onClick={() => { setPlaying(false); setStep((s) => Math.max(0, s - 1)); }}>◀ Step</button>
              <button onClick={() => { setPlaying(false); setStep((s) => Math.min(timeline.length - 1, s + 1)); }}>Step ▶</button>
            </div>
            <div className="slider">
              <label><span>Speed</span><span>{speed} steps/s</span></label>
              <input type="range" min={1} max={30} value={speed}
                     onChange={(e) => setSpeed(+e.target.value)} />
            </div>
            <div className="slider">
              <label><span>Timeline</span><span>{step} / {timeline.length - 1}</span></label>
              <input type="range" min={0} max={timeline.length - 1} value={step}
                     onChange={(e) => { setPlaying(false); setStep(+e.target.value); }} />
            </div>
          </div>
        </div>

        <div className="card">
          <h2>Legend</h2>
          <div className="legend">
            <div className="item"><span className="dot" style={{ background: "#4f8cff" }} /> Query (transform 45°)</div>
            <div className="item"><span className="dot" style={{ background: "#ffd166" }} /> Entry point</div>
            <div className="item"><span className="dot" style={{ background: "#ff5c5c" }} /> Current node</div>
            <div className="item"><span className="dot" style={{ background: "#cdd6e4" }} /> Visited</div>
            <div className="item"><span className="dot" style={{ background: "#3ddc97" }} /> Top-k result</div>
            <div className="item"><span className="swatch" style={{ background: "#ffd166" }} /> Descent path</div>
            <div className="item"><span className="swatch" style={{ background: "#4f8cff" }} /> Traversal edges</div>
          </div>
        </div>
      </div>
    </div>
  );
}
