"""RAG Q2: the architecture of a basic RAG system: an offline indexing lane and an online query lane."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 470, "A basic RAG system", "Indexing runs offline and writes vectors; every query reads them back, then the model answers from what was retrieved.")
# offline lane
s.region(24, 80, 852, 140, "OFFLINE · indexing", "data")
x = 60
for label, sub in [("Sources", "PDFs, wikis, tickets"), ("Parse + clean", "text, tables, metadata"), ("Chunk", "~300–800 tokens, overlap"), ("Embed", "embedding model")]:
    s.box(x, 118, 150, 70, label, [sub], "data", size=13.5)
    s.arrow([(x + 150, 153), (x + 172, 153)], "data")
    x += 172
s.cylinder(x + 60, 153, 120, 84, "Vector index", "data")
s.text(x + 48, 204, "+ BM25 keyword index", size=11, fill="#0A5A51", anchor="end")

# online lane
s.region(24, 244, 852, 206, "ONLINE · per query", "compute")
s.pill(100, 320, 130, 40, "User query", "human")
s.arrow([(165, 320), (196, 320)], "compute")
s.box(198, 290, 150, 60, "Embed query", ["same model as index"], "compute", size=13.5)
s.arrow([(348, 320), (380, 320)], "compute")
s.box(382, 290, 150, 60, "Retrieve top-k", ["ANN + filters (ACL)"], "compute", size=13.5)
s.arrow([(x + 60, 196), (x + 60, 250), (457, 250), (457, 288)], "data", dashed=True, label="similarity search", label_at=0.55, label_dy=-9)
s.arrow([(532, 320), (564, 320)], "compute")
s.box(566, 290, 150, 60, "Rerank", ["cross-encoder"], "amber", size=13.5)
s.arrow([(641, 350), (641, 380)], "amber")
# second row runs right to left
s.box(530, 382, 300, 58, "Prompt = instructions + chunks + query", ["cite sources; say 'not found' if absent"], "slate", size=12.5)
s.arrow([(530, 411), (478, 411)], "model")
s.box(330, 382, 146, 58, "LLM", ["answers from the chunks"], "model", size=14)
s.arrow([(330, 411), (268, 411)], "output")
s.pill(180, 411, 172, 40, "Answer + citations", "output", size=12.5)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/03-retrieval-augmented-generation-rag/q02-basic-rag.svg")
