"""Multimodal Q26: multi-page documents as retrieve, map (per page at full resolution), reduce."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 470, "Multi-page documents: retrieve, map, reduce", "Find the relevant pages, read each at full resolution, extract per page, then reason over the extracted text.")
# a stack of pages
L = PALETTE["slate"]
for i in range(5):
    x, y = 36 + i * 6, 176 + i * 6
    s.add(f'<rect x="{x}" y="{y}" width="92" height="118" rx="5" fill="#FFFFFF" stroke="{L[0]}" stroke-width="1.3"/>')
fx, fy = 60, 200
for k in range(7):
    w = 70 if k % 3 else 48
    s.add(f'<rect x="{fx + 11}" y="{fy + 14 + k * 13}" width="{w}" height="5" rx="2" fill="#D0D5DD"/>')
s.text(106, 346, "Multi-page document", size=12, weight=700, fill=L[2])
s.text(106, 363, "text layer or OCR,", size=11, fill=L[2])
s.text(106, 378, "+ page images", size=11, fill=L[2])

s.arrow([(156, 258), (198, 258)], "slate")
s.box(200, 222, 150, 72, "Retrieve pages", ["text search or", "ColPali-style"], "compute", size=14)

s.text(275, 108, "retrieve", size=11.5, weight=700, fill=PALETTE["compute"][2])
s.text(562, 108, "map: one page per call", size=11.5, weight=700, fill=PALETTE["model"][2])
s.text(823, 108, "reduce", size=11.5, weight=700, fill=PALETTE["output"][2])

ys = [160, 258, 356]
titles = ["Page 7 of 40", "Page … of 40", "Page … of 40"]
for y, t in zip(ys, titles):
    s.arrow([(350, 258), (393, y)], "model")
    s.pill(470, y, 150, 38, "VLM, full res", "model", size=12.5)
    s.arrow([(545, y), (578, y)], "model")
    s.box(580, y - 28, 150, 56, t, ["extract to schema"], "data", size=13.5)
    s.arrow([(730, y), (768, 258 + (y - 258) * 0.24)], "output")
s.box(770, 222, 106, 72, "Aggregate", ["in text or code"], "output", size=13.5, detail=11)
s.arrow([(823, 294), (823, 380)], "output")
s.box(760, 382, 116, 56, "Answer", ["page citations"], "output", size=14, detail=11)

s.text(36, 446, "Measure page retrieval recall separately from answer accuracy.", size=11.5, fill="#344054", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/11-multimodal-ai/q26-multipage-documents.svg")
