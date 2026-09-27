"""Vector DBs Q18: billion-scale vector search: compressed shards, scatter-gather, then rescoring."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 512, "Billions of vectors: compress, shard, scatter-gather", "Memory comes first. Compressed codes in RAM, shards queried in parallel, and full vectors only for the final rescore.")
C = 290
s.pill(C, 102, 110, 36, "Query", "human")
s.arrow([(C, 120), (C, 138)], "compute")
s.box(C - 100, 140, 200, 50, "Coordinator", ["scatter to shards"], "compute", size=13.5)
xs = [110, 290, 470]
for x, name in zip(xs, ["Shard 1", "Shard 2", "Shard N"]):
    s.arrow([(C + (x - C) * 0.25, 190), (x, 226)], "compute")
    s.box(x - 76, 228, 152, 70, name, ["PQ codes in RAM,", "full vectors on SSD"], "data", size=13.5)
    s.arrow([(x, 298), (C + (x - C) * 0.3, 334)], "data")
s.text(380, 263, "···", size=16, weight=700, fill="#0A5A51")
s.pill(C, 352, 200, 36, "Merge candidates", "data", size=12.5)
s.arrow([(C, 370), (C, 392)], "amber")
s.box(C - 110, 394, 220, 48, "Rescore with full vectors", role="amber", size=13)
s.arrow([(C, 442), (C, 460)], "output")
s.pill(C, 476, 100, 30, "Top-k", "output", size=12.5)

# memory math and sharding choice
s.region(578, 84, 280, 400, "Memory first: 1B vectors", "slate", dashed=False)
st, _, ink = PALETTE["slate"]
s.text(596, 128, "float32, 768 dims", size=11.5, fill=ink, weight=600, anchor="start")
s.add(f'<rect x="596" y="138" width="244" height="22" rx="4" fill="{st}" fill-opacity="0.85"/>')
s.text(832, 149, "≈ 3 TB", size=12, fill="#FFFFFF", weight=700, anchor="end")
st, _, ink = PALETTE["output"]
s.text(596, 184, "PQ, 64 bytes per vector", size=11.5, fill=ink, weight=600, anchor="start")
s.add(f'<rect x="596" y="194" width="6" height="22" rx="2" fill="{st}" fill-opacity="0.85"/>')
s.text(612, 205, "≈ 64 GB of codes", size=12, fill=ink, weight=700, anchor="start")

s.text(596, 250, "Where does a query go?", size=12.5, fill="#344054", weight=700, anchor="start")
s.box(596, 264, 244, 58, "Hash sharding", ["balanced, but every query", "hits every shard"], "compute", size=13, detail=11)
s.box(596, 332, 244, 58, "Shard by attribute", ["tenant, region, language:", "only the relevant shards"], "output", size=13, detail=11)
s.text(596, 418, "Shards add capacity;", size=11.5, fill="#344054", anchor="start")
s.text(596, 435, "replicas add query throughput.", size=11.5, fill="#344054", anchor="start")
s.text(596, 458, "Do the math before buying hardware.", size=11.5, fill="#8E2A23", weight=600, anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/06-vector-databases-and-embeddings/q18-billion-scale.svg")
