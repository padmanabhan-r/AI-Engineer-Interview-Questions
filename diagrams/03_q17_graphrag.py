"""RAG Q17: GraphRAG, an LLM-built entity graph with community summaries, searched locally or globally."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 470, "GraphRAG: retrieve over a graph the LLM built", "Indexing extracts entities and relations, groups them into communities and summarizes each; queries search locally or globally.")
s.region(24, 80, 832, 170, "INDEXING · one or more LLM calls per chunk", "data")
s.cylinder(92, 172, 100, 76, "Chunks", "data")
s.arrow([(142, 172), (160, 172)], "data")
s.box(162, 132, 150, 80, "LLM extracts", ["entities, relations,", "claims per chunk"], "model", size=13.5)
s.arrow([(312, 172), (332, 172)], "model")

# the same small graph twice: plain, then coloured by community
N = [(30, 30), (70, 18), (58, 58), (100, 44), (120, 76), (26, 76), (138, 30), (86, 84)]
E = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 5), (3, 6), (3, 4), (4, 7), (6, 4), (2, 7), (0, 5)]
def graph(x0, y0, title, colours=None, hulls=()):
    s.add(f'<rect x="{x0}" y="{y0}" width="170" height="126" rx="12" fill="#FFFFFF" stroke="{PALETTE["data"][0]}" stroke-width="1.6"/>')
    s.text(x0 + 85, y0 + 18, title, size=13, weight=700, fill=PALETTE["data"][2])
    for cx, cy, r, role in hulls:
        s.add(f'<circle cx="{x0 + 12 + cx}" cy="{y0 + 24 + cy}" r="{r}" fill="{PALETTE[role][1]}" stroke="{PALETTE[role][0]}" stroke-dasharray="4 3" stroke-width="1.2"/>')
    for a, b in E:
        (ax, ay), (bx, by) = N[a], N[b]
        s.add(f'<line x1="{x0 + 12 + ax}" y1="{y0 + 24 + ay}" x2="{x0 + 12 + bx}" y2="{y0 + 24 + by}" stroke="#98A2B3" stroke-width="1.4"/>')
    for i, (nx, ny) in enumerate(N):
        c = PALETTE[colours[i]][0] if colours else PALETTE["data"][0]
        s.add(f'<circle cx="{x0 + 12 + nx}" cy="{y0 + 24 + ny}" r="6.5" fill="{c}" stroke="#FFFFFF" stroke-width="1.5"/>')

graph(334, 109, "Entity graph")
s.arrow([(504, 172), (524, 172)], "data")
graph(526, 109, "Leiden communities",
      ["model", "model", "model", "amber", "amber", "model", "amber", "pink"],
      hulls=[(40, 42, 40, "model"), (118, 50, 36, "amber")])
s.arrow([(696, 172), (716, 172)], "data")
s.cylinder(782, 172, 124, 86, "Community\nsummaries", "data", size=12.5)

# query lane
s.region(24, 272, 832, 176, "QUERY", "compute")
s.pill(112, 352, 130, 42, "Query", "human")
s.arrow([(177, 352), (330, 352)], "compute")
s.box(332, 316, 190, 72, "Local search", ["matched entities, their", "neighbourhood, source chunks"], "compute", size=13.5)
s.arrow([(112, 373), (112, 424), (752, 424), (752, 390)], "compute")
s.box(652, 306, 196, 84, "Global search", ["map-reduce over community", "summaries: \"main risk themes", "across 5,000 reports\""], "compute", size=13.5)
s.arrow([(419, 235), (419, 314)], "data", dashed=True)
s.arrow([(782, 215), (782, 304)], "data", dashed=True)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/03-retrieval-augmented-generation-rag/q17-graphrag.svg")
