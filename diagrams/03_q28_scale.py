"""RAG Q28: retrieval at millions of documents: route, scatter to compressed shards, gather, rescore, rerank."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(880, 470, "Scaling RAG to millions of documents", "Partition before you shard; compress to fit in RAM, then let rescoring and reranking recover what the lossy first stage lost.")
s.pill(76, 210, 100, 40, "Query", "human")
s.arrow([(126, 210), (148, 210)], "human")
s.box(150, 170, 140, 80, "Router", ["tenant and", "metadata filters"], "compute", size=13.5)
s.region(318, 86, 214, 250, "scatter-gather", "data", label_pos="tl")
s.arrow([(290, 210), (304, 210), (304, 150), (348, 150)], "compute")
s.arrow([(304, 210), (304, 282), (348, 282)], "compute")
s.cylinder(425, 150, 150, 64, "Shard 1", "data")
s.text(425, 197, "⋮", size=18, fill="#0A5A51", weight=700)
s.cylinder(425, 282, 150, 64, "Shard N", "data")
s.text(425, 322, "HNSW · int8 / PQ · replicas for QPS", size=10.5, fill="#0A5A51")
s.arrow([(500, 150), (548, 150), (548, 216), (578, 216)], "data")
s.arrow([(500, 282), (548, 282), (548, 216)], "data", head=False)
s.box(580, 186, 140, 60, "Merge", ["top candidates"], "data", size=13.5)
s.arrow([(650, 246), (650, 368)], "data")
# second row, right to left
s.box(580, 370, 140, 60, "Rescore", ["full precision"], "amber", size=13.5)
s.arrow([(580, 400), (536, 400)], "amber")
s.box(376, 370, 158, 60, "Rerank", ["cross-encoder"], "amber", size=13.5)
s.arrow([(376, 400), (330, 400)], "amber")
s.box(210, 370, 118, 60, "LLM", ["answers"], "model", size=14)

# the arithmetic that forces this design
s.box(744, 86, 118, 250, "Memory", ["", "50M chunks", "× 768 dims", "× 4 bytes", "≈ 154 GB float32", "before graph", "overhead", "", "int8: 4× smaller", "binary: 32×"], "slate", size=13.5, detail=11)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/03-retrieval-augmented-generation-rag/q28-scale.svg")
