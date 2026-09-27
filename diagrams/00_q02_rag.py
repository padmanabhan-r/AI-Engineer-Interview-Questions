"""Must Know Q2: RAG in one picture: build an index offline, retrieve and rerank at query time, answer with citations."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(880, 470, "RAG: answer from your own data, fetched at query time", "The model answers from supplied evidence instead of memory, and cites the chunks it used.")
# offline lane
s.region(24, 80, 832, 128, "OFFLINE · build the index", "data")
s.box(48, 118, 140, 62, "Documents", ["your own data"], "data", size=13.5)
s.arrow([(188, 149), (208, 149)], "data")
s.box(210, 118, 140, 62, "Chunk", ["split into passages"], "data", size=13.5)
s.arrow([(350, 149), (370, 149)], "data")
s.box(372, 118, 150, 62, "Embed", ["one vector per chunk"], "data", size=13.5)
s.arrow([(522, 149), (542, 149)], "data")
s.cylinder(610, 149, 130, 76, "Index", "data")
s.text(690, 140, "vectors + chunk text", size=11.5, fill="#0A5A51", anchor="start")
s.text(690, 158, "often BM25 alongside", size=11.5, fill="#0A5A51", anchor="start")

# online lane
s.region(24, 226, 832, 222, "ONLINE · per question", "compute")
s.pill(118, 296, 140, 40, "Question", "human")
s.arrow([(188, 296), (250, 296)], "compute")
s.box(252, 266, 160, 60, "Embed question", ["query → vector"], "compute", size=13.5)
s.arrow([(412, 296), (543, 296)], "compute")
s.box(545, 266, 130, 60, "Retrieve top-k", ["nearest chunks"], "compute", size=13.5)
s.arrow([(610, 187), (610, 264)], "data", dashed=True)
s.arrow([(675, 296), (698, 296)], "compute")
s.box(700, 266, 140, 60, "Rerank", ["cross-encoder"], "amber", size=13.5)
s.arrow([(770, 326), (770, 360)], "amber")
# second row runs right to left
s.box(620, 362, 220, 62, "Prompt", ["instructions + chunks + question"], "slate", size=13.5)
s.arrow([(620, 393), (580, 393)], "model")
s.box(428, 362, 150, 62, "LLM", ["answers from evidence"], "model", size=14)
s.arrow([(428, 393), (404, 393)], "output")
s.pill(314, 393, 176, 40, "Answer + chunk IDs", "output", size=12.5)
s.box(44, 358, 164, 70, "Measure apart", ["retrieval: recall@k", "answer: faithfulness"], "pink", size=13)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/00-must-know/q02-rag.svg")
