"""Behavioral Q7: debugging a poor RAG system: trace each failure to the stage where the right chunk was lost."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 540, "Debugging a poor RAG system: find where it broke", "Stop tuning prompts. Trace each failing query to the stage that lost the right chunk, fix the largest class, re-measure.")

C = [140, 353, 566, 779]
# the funnel: each stage keeps fewer chunks
stages = [("Corpus", "data", 56), ("Top 50 retrieved", "compute", 48), ("Final top-k", "amber", 40), ("Answer", "model", 32)]
for cx, (label, role, h) in zip(C, stages):
    col, fill, ink = PALETTE[role]
    x0, x1, y = cx - 104, cx + 104, 122
    pts = f"{x0},{y - h/2} {x1 - 14},{y - h/2} {x1},{y} {x1 - 14},{y + h/2} {x0},{y + h/2} {x0 + 14},{y}"
    s.add(f'<polygon points="{pts}" fill="{fill}" stroke="{col}" stroke-width="1.6"/>')
    s.text(cx + 4, y, label, size=12.5, weight=700, fill=ink)

# the questions, left to right
qs = ["Answer in\ncorpus?", "Right chunk\nin top 50?", "In final\ntop-k?"]
for cx, q in zip(C, qs):
    s.arrow([(cx, 152), (cx, 170)], "slate", width=1.5)
    s.diamond(cx, 214, 172, 84, q, "amber", size=12)
for a, b in zip(C, C[1:]):
    s.arrow([(a + 86, 214), (b - 88, 214)], "output")
    s.text((a + b) / 2, 202, "yes", size=11.5, fill="#275C1C", weight=600)
s.arrow([(C[3], 150), (C[3], 170)], "slate", width=1.5)
s.box(C[3] - 86, 176, 186, 76, "Generation", ["prompt, context order,", "model"], "model", size=13.5)

# the fixes, when the answer is no
fixes = [("Content gap", ["test abstention"], "fail"), ("Retrieval", ["chunking, hybrid search,", "filters"], "compute"),
         ("Reranker", ["add or tune one"], "amber")]
for cx, (title, lines, role) in zip(C, fixes):
    s.arrow([(cx, 256), (cx, 290)], "fail")
    s.text(cx + 10, 273, "no", size=11.5, fill="#8E2A23", weight=600, anchor="start")
    s.box(cx - 100, 292, 200, 70, title, lines, role, size=13.5)
s.text(C[3] + 7, 300, "typical fixes", size=11, fill="#667085", weight=700)
s.text(C[3] + 7, 318, "structure-aware chunking for tables", size=11, fill="#667085")
s.text(C[3] + 7, 334, "BM25 for exact identifiers", size=11, fill="#667085")
s.text(C[3] + 7, 350, "a reranker · version filters", size=11, fill="#667085")

# measurement
s.region(30, 390, 840, 128, "", "slate", dashed=False)
s.text(52, 416, "One number splits retrieval from generation", size=13, weight=700, fill="#344054", anchor="start")
s.text(52, 440, "Label the correct source for 50–100 failures,", size=11.5, fill="#344054", anchor="start")
s.text(52, 458, "then measure recall@k at k = 5, 20 and 50.", size=11.5, fill="#344054", anchor="start")
s.text(52, 490, "Never change embeddings, chunk size and prompt at once.", size=11.5, fill="#8E2A23", weight=600, anchor="start")
base = 496
for i, (k, h) in enumerate([("@5", 34), ("@20", 52), ("@50", 66)]):
    x = 470 + i * 58
    s.add(f'<rect x="{x}" y="{base - h}" width="36" height="{h}" rx="4" fill="{PALETTE["compute"][0]}" fill-opacity="{0.45 + 0.2 * i}"/>')
    s.text(x + 18, base + 12, k, size=11, fill="#1B418C", mono=True)
s.add(f'<line x1="462" y1="{base}" x2="640" y2="{base}" stroke="#98A2B3"/>')
s.text(660, 440, "read it against the tree:", size=11, fill="#667085", anchor="start")
s.text(660, 456, "low at 50 → retrieval;", size=11, fill="#667085", anchor="start")
s.text(660, 472, "high at 50, low at k → rerank;", size=11, fill="#667085", anchor="start")
s.text(660, 488, "high at k → generation", size=11, fill="#667085", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/14-behavioral-and-scenario-based-questions/q07-rag-debugging.svg")
