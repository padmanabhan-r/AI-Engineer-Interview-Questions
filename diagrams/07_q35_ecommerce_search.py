"""System design Q35: e-commerce search as hybrid retrieval plus learning-to-rank, with LLMs kept offline and on the long tail."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 530, "E-commerce search", "Lexical and semantic retrieval, a learned ranker on conversion signals; LLMs enrich offline, not on every request.")

# offline lane
s.region(24, 78, 852, 122, "OFFLINE · once per product change", "data")
s.cylinder(92, 150, 124, 70, "Catalogue\n20M SKUs", "data", size=12.5)
s.arrow([(154, 150), (178, 150)], "data")
s.box(180, 116, 210, 68, "LLM enrichment", ["normalise attributes,", "fill gaps"], "model", size=13.5)
s.arrow([(390, 150), (402, 150)], "data")
s.cylinder(472, 150, 136, 72, "BM25 fields\n+ vectors", "data", size=12.5)
s.text(566, 136, "20M × 768-dim int8 ≈ 15 GB", size=11.5, fill="#0A5A51", anchor="start", weight=600)
s.text(566, 158, "in memory on a few replicated nodes", size=11.5, fill="#0A5A51", anchor="start")

# online lane
s.region(24, 218, 852, 290, "ONLINE · per query, p99 < ~200 ms", "compute")
s.pill(84, 322, 104, 40, "Query", "human")
s.arrow([(136, 322), (156, 322)], "compute")
s.box(158, 284, 190, 76, "Query understanding", ["spelling, attributes,", "category → filters"], "compute", size=13.5)
s.arrow([(253, 360), (253, 404)], "model", dashed=True)
s.text(261, 382, "long tail only", size=11, fill="#3B2596", weight=600, anchor="start")
s.box(158, 406, 190, 58, "Long-tail LLM", ["cached per normalised query"], "model", size=13)

s.region(384, 250, 176, 164, "hybrid", "slate", label_pos="bl")
s.arrow([(472, 186), (472, 262)], "data", dashed=True)
s.text(482, 232, "serves both", size=11.5, fill="#0A5A51", weight=600, anchor="start")
s.box(398, 264, 150, 52, "BM25", ["brands, SKUs, IDs"], "compute", size=13)
s.box(398, 328, 150, 52, "Embeddings", ["ANN, vague intent"], "compute", size=13)
s.arrow([(348, 322), (370, 322), (370, 290), (396, 290)], "compute")
s.arrow([(370, 322), (370, 354), (396, 354)], "compute")

s.arrow([(548, 290), (590, 290)], "compute")
s.arrow([(548, 354), (590, 354)], "compute")
s.box(592, 272, 118, 100, "Merge", ["filter, in-stock"], "compute", size=13.5)
s.arrow([(710, 322), (728, 322)], "model")
s.box(730, 272, 140, 100, "Ranker", ["learning-to-rank:", "CTR, conversion,", "stock, affinity"], "model", size=13.5)
s.arrow([(800, 372), (800, 430)], "output")
s.pill(794, 452, 156, 40, "Results + facets", "output", size=12)

s.arrow([(716, 452), (672, 452)], "data")
s.cylinder(604, 452, 132, 66, "Clicks, carts,\npurchases", "data", size=12)
s.arrow([(620, 419), (620, 398), (748, 398), (748, 374)], "data", dashed=True)
s.text(530, 452, "trains the ranker,\nposition-bias corrected", size=11, fill="#0A5A51", weight=600, anchor="end")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q35-ecommerce-search.svg")
