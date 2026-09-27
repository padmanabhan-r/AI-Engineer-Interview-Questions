"""System design Q9: multimodal search, every item indexed twice (vectors and derived text), fused into moments."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 540, "Multimodal search over images and video", "Index every item as vectors and as text, search both, then fuse and rerank so results land on moments, not files.")

s.region(24, 312, 852, 208, "QUERY · p95 < ~300 ms", "compute", label_pos="bl")
s.region(24, 76, 852, 222, "INDEX · every item twice", "data")

# ingest
s.pill(90, 200, 116, 50, "Images +\nvideos", "human", size=12.5)
s.arrow([(148, 190), (158, 190), (158, 146), (170, 146)], "data")
s.arrow([(148, 212), (158, 212), (158, 256), (170, 256)], "data")
s.box(172, 110, 164, 72, "Shot detection", ["keyframe per ~5 s"], "data", size=13.5)
s.box(172, 228, 164, 56, "ASR", ["speech transcripts"], "data", size=13.5)

# two representations
s.arrow([(336, 134), (358, 134), (358, 118), (380, 118)], "model")
s.arrow([(336, 158), (358, 158), (358, 191), (380, 191)], "compute")
s.box(382, 92, 184, 54, "Image-text embed", ["CLIP / SigLIP-style"], "model", size=13.5)
s.box(382, 166, 184, 50, "Captions + OCR", [], "compute", size=13.5)
s.arrow([(566, 118), (594, 118)], "model")
s.arrow([(566, 191), (580, 191), (580, 214), (594, 214)], "compute")
s.arrow([(336, 256), (594, 256)], "compute")

s.cylinder(670, 118, 150, 66, "", "model")
s.text(670, 116, "Vector index", size=13, weight=600, fill="#3B2596")
s.text(670, 136, "~770M vectors", size=11, fill="#3B2596")
s.cylinder(670, 238, 150, 90, "Text index", "compute")

# query
s.pill(126, 380, 188, 42, "Text or image query", "human", size=12.5)
s.arrow([(220, 380), (578, 380)], "compute", label="search both indexes", label_dy=-11)
s.box(580, 350, 176, 62, "Fuse", ["group by video + time"], "amber", size=14)
s.arrow([(670, 283), (670, 348)], "compute")
s.arrow([(745, 118), (800, 118), (800, 381), (758, 381)], "model")
s.arrow([(668, 412), (668, 460), (562, 460)], "amber")
s.box(390, 430, 170, 60, "Rerank top 50", ["vision-language model"], "amber", size=13.5)
s.arrow([(390, 460), (324, 460)], "output")
s.pill(222, 460, 200, 42, "Moments + timestamps", "output", size=12.5)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q09-multimodal-search.svg")
