"""Vector DBs Q7: ANN search with an IVF index: compare with centroids, search only the nprobe nearest cells."""
import sys, random, math; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 486, "ANN search: look only where the answer probably is", "An IVF index clusters the vectors ahead of time; a query scans the nprobe nearest cells, not all of them.")

# the vector space, partitioned into cells
s.region(24, 80, 440, 362, "vector space, k-means cells", "slate", dashed=False)
cents = {1: (80, 160), 2: (190, 150), 3: (330, 160), 4: (74, 272), 5: (180, 256), 6: (420, 270), 7: (300, 318), 8: (150, 364), 9: (410, 364)}
q = (330, 238)
probe = {3, 7}
rng = random.Random(7)
for k, (cx, cy) in cents.items():
    role = "data" if k in probe else "slate"
    st, fi, ink = PALETTE[role]
    s.add(f'<circle cx="{cx}" cy="{cy}" r="36" fill="{fi}" fill-opacity="{0.9 if k in probe else 0.6}" stroke="{st}" stroke-opacity="{0.8 if k in probe else 0.35}" stroke-width="1.4"/>')
for k, (cx, cy) in cents.items():
    st = PALETTE["data" if k in probe else "slate"][0]
    for _ in range(9):
        a, r = rng.uniform(0, 2 * math.pi), rng.uniform(8, 28)
        s.add(f'<circle cx="{cx + r * math.cos(a):.1f}" cy="{cy + r * math.sin(a):.1f}" r="2.6" fill="{st}" fill-opacity="{0.9 if k in probe else 0.45}"/>')
# query to every centroid (dashed), nearest two solid
for k, (cx, cy) in cents.items():
    if k in probe:
        continue
    s.add(f'<line x1="{q[0]}" y1="{q[1]}" x2="{cx}" y2="{cy}" stroke="#D6408F" stroke-width="1" stroke-dasharray="3 4" stroke-opacity="0.55"/>')
for k in probe:
    cx, cy = cents[k]
    s.add(f'<line x1="{q[0]}" y1="{q[1]}" x2="{cx}" y2="{cy}" stroke="#D6408F" stroke-width="2.2"/>')
for k, (cx, cy) in cents.items():
    st, _, ink = PALETTE["data" if k in probe else "slate"]
    s.add(f'<rect x="{cx - 6}" y="{cy - 6}" width="12" height="12" transform="rotate(45 {cx} {cy})" fill="#FFFFFF" stroke="{st}" stroke-width="2"/>')
    dx = -18 if k == 7 else 18
    s.text(cx + dx, cy - 18, str(k), size=12, weight=700, fill=ink)
s.add(f'<circle cx="{q[0]}" cy="{q[1]}" r="8" fill="#D6408F" stroke="#FFFFFF" stroke-width="2.5"/>')
s.text(q[0] + 12, q[1] - 16, "query", size=12, weight=700, fill="#8A1F58", anchor="start")
s.text(40, 424, "◇ centroid, dots are vectors; teal = the nprobe = 2 cells searched", size=11, fill="#667085", anchor="start")

# the steps
X = 672
s.pill(X, 104, 120, 36, "Query", "pink")
s.arrow([(X, 122), (X, 144)], "pink")
s.box(X - 110, 146, 220, 58, "Compare with centroids", ["one distance per centroid"], "compute", size=13.5)
s.arrow([(X - 20, 204), (X - 66, 236)], "data")
s.arrow([(X + 20, 204), (X + 66, 236)], "data")
s.pill(X - 90, 258, 140, 40, "Search cell 3", "data", size=12.5)
s.pill(X + 90, 258, 140, 40, "Search cell 7", "data", size=12.5)
s.arrow([(X - 66, 278), (X - 20, 308)], "data")
s.arrow([(X + 66, 278), (X + 20, 308)], "data")
s.pill(X, 328, 190, 40, "Merge candidates", "compute", size=12.5)
s.arrow([(X, 348), (X, 370)], "output")
s.pill(X, 390, 110, 38, "Top-k", "output")
s.text(X, 432, "knob: nprobe for IVF, efSearch for HNSW", size=11.5, fill="#344054", weight=600)
s.text(452, 466, "Exact search over 100M × 768 vectors is about 77 billion multiply-adds per query.", size=11.5, fill="#667085", italic=True)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/06-vector-databases-and-embeddings/q07-ann-ivf.svg")
