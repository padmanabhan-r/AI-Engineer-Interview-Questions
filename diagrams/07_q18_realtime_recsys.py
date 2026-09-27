"""System design Q18: a real-time recommender: online funnel, streaming feature loop, offline content understanding."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 514, "A real-time recommendation system", "Retrieve hundreds, rank with fresh features, re-rank for diversity; LLMs work offline on items, not in the request path.")
# online funnel
s.region(24, 80, 852, 128, "ONLINE · per request, p99 under ~100 ms", "compute")
s.pill(82, 150, 100, 40, "Request", "human")
s.arrow([(132, 150), (158, 150)], "compute")
s.box(160, 114, 220, 72, "Candidates", ["two-tower ANN, trending, graph", "→ hundreds of items"], "compute", size=13.5)
s.arrow([(380, 150), (408, 150)], "compute")
s.box(410, 114, 190, 72, "Multi-task ranker", ["click, completion, like,", "skip, report"], "model", size=13.5)
s.arrow([(600, 150), (628, 150)], "compute")
s.box(630, 120, 140, 60, "Re-rank", ["diversity, rules"], "amber", size=13.5)
s.arrow([(770, 150), (792, 150)], "output")
s.pill(834, 150, 80, 40, "Feed", "output")

# streaming loop
s.region(386, 214, 490, 148, "STREAMING · reacts in seconds", "data", label_pos="br")
s.arrow([(834, 170), (834, 282)], "output")
s.text(824, 250, "clicks, skips,\nhides", size=11, fill="#275C1C", weight=600, anchor="end")
s.pill(830, 304, 90, 40, "Events", "data", size=12.5)
s.arrow([(785, 304), (752, 304)], "data")
s.box(570, 276, 180, 56, "Stream processor", ["~2.5B events/day"], "data", size=13)
s.arrow([(570, 304), (556, 304)], "data")
s.cylinder(478, 304, 150, 64, "Online\nfeature store", "data", size=12.5)
s.arrow([(505, 273), (505, 188)], "data")
s.text(514, 200, "real-time features", size=11.5, fill="#0A5A51", weight=600, anchor="start")

# offline: content understanding writes the item vectors
s.region(24, 372, 852, 124, "OFFLINE · content understanding", "model", label_pos="tr")
s.cylinder(270, 430, 170, 76, "Item ANN index", "compute", size=12.5)
s.text(270, 482, "item vectors, precomputed", size=11, fill="#1B418C")
s.arrow([(270, 394), (270, 188)], "compute")
s.box(390, 400, 280, 64, "LLM + vision", ["embeddings and tags per item", "help cold start"], "model", size=13.5)
s.arrow([(390, 432), (357, 432)], "model")
s.arrow([(724, 432), (672, 432)], "model")
s.pill(790, 432, 130, 40, "New items", "human")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q18-realtime-recsys.svg")
