"""System design Q3: chat with your documents, an ingest lane and a query lane with hybrid search and rerank."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 500, "Chat with your documents", "Most of the quality lives in parsing and retrieval; the model only answers from the passages it is handed.")

s.region(24, 262, 852, 218, "QUERY · per question", "compute", label_pos="bl")
# ingest lane
s.region(24, 80, 852, 150, "INGEST · searchable within minutes", "data")
s.pill(100, 158, 120, 42, "Upload", "human")
s.arrow([(160, 158), (190, 158)], "data")
s.box(192, 118, 200, 80, "Layout-aware parse", ["tables stay tables,", "headers dropped"], "data", size=13.5)
s.arrow([(392, 158), (422, 158)], "data")
s.box(424, 118, 200, 80, "Structural chunks", ["300–800 tokens, with", "title + section path"], "data", size=13.5)
s.arrow([(624, 158), (660, 158)], "data")
s.cylinder(752, 150, 170, 92, "Vector + BM25", "data")
s.text(772, 214, "~600k chunks · ~2.4 GB", size=11, fill="#0A5A51")

# query lane, first row left to right
s.pill(126, 320, 180, 42, "Question + history", "human", size=12.5)
s.arrow([(216, 320), (240, 320)], "compute")
s.box(242, 290, 178, 60, "Rewrite", ["to a standalone query"], "compute", size=13.5)
s.arrow([(420, 320), (450, 320)], "compute")
s.box(452, 290, 200, 60, "Hybrid search", ["dense + BM25, RRF fusion"], "compute", size=13.5)
s.arrow([(690, 190), (690, 240), (600, 240), (600, 288)], "data", dashed=True)
s.arrow([(652, 320), (688, 320)], "compute")
s.box(690, 290, 166, 60, "Rerank", ["cross-encoder on 50"], "amber", size=13.5)

# second row right to left
s.arrow([(773, 350), (773, 386)], "amber", label="", head=True)
s.text(783, 368, "top 5–8", size=11.5, fill="#7A5300", weight=600, anchor="start")
s.box(560, 388, 296, 62, "LLM answers from passages", ["~4k context tokens"], "model", size=13.5)
s.arrow([(560, 419), (472, 419)], "output")
s.pill(360, 419, 220, 40, "Answer with citations", "output", size=12.5)
s.text(360, 456, "or \"not in your documents\"", size=11.5, fill="#275C1C", weight=600)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q03-rag-system.svg")
