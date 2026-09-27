"""RAG Q19: route structured questions to SQL; retrieve schema and definitions, not rows."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 420, "Structured data: route to a query engine", "Aggregations and exact figures need every matching row, so the database computes and the LLM explains.")
s.pill(80, 240, 110, 42, "Question", "human")
s.arrow([(135, 240), (158, 240)], "human")
s.diamond(225, 240, 130, 100, "Documents\nor data?", "amber", size=12)
# documents lane
s.arrow([(225, 190), (225, 138), (318, 138)], "amber")
s.text(215, 164, "documents", size=11.5, fill="#7A5300", weight=600, anchor="end")
s.box(320, 108, 190, 60, "Vector + keyword RAG", ["chunks, tables as Markdown"], "data", size=13)
s.arrow([(510, 138), (578, 138)], "data")
s.box(580, 102, 160, 72, "LLM answers", ["from the chunks", "or the rows"], "model", size=13.5)
s.arrow([(740, 138), (764, 138)], "output")
s.box(766, 108, 116, 60, "Answer", ["+ the SQL used"], "output", size=13.5)
# data lane
s.arrow([(225, 290), (225, 338), (268, 338)], "amber")
s.text(215, 314, "data", size=11.5, fill="#7A5300", weight=600, anchor="end")
s.box(270, 306, 168, 64, "Retrieve context", ["tables, definitions,", "verified SQL examples"], "compute", size=13)
s.arrow([(438, 338), (460, 338)], "compute")
s.box(462, 310, 124, 56, "LLM writes SQL", [], "model", size=13)
s.arrow([(586, 338), (606, 338)], "model")
s.hexagon(680, 338, 144, 66, "Validate", "amber")
s.text(680, 382, "read-only · no DDL/DML · LIMIT", size=10.5, fill="#7A5300", mono=True)
s.arrow([(752, 338), (768, 338)], "amber")
s.box(770, 310, 108, 56, "Execute", ["as the user"], "data", size=13)
s.arrow([(824, 310), (824, 262), (660, 262), (660, 176)], "data", label="rows", label_at=0.55, label_dy=-10)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/03-retrieval-augmented-generation-rag/q19-structured-data.svg")
