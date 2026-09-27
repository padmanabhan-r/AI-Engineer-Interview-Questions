"""RAG Q10: two-stage retrieval, a fast retriever for recall and a cross-encoder for precision."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 400, "Re-ranking: recall first, precision second", "A cheap retriever casts a wide net; a cross-encoder reads each pair together and keeps only the best few.")
s.pill(78, 150, 100, 40, "Query", "human")
s.arrow([(128, 150), (156, 150)], "human")
s.box(158, 116, 170, 68, "Hybrid retrieval", ["fast, tuned for recall"], "compute", size=13.5)
s.arrow([(328, 150), (358, 150)], "compute")
s.box(360, 108, 250, 84, "Cross-encoder", ["[CLS] query [SEP] passage [SEP]", "full attention → one score"], "amber", size=13.5)
s.arrow([(610, 150), (640, 150)], "amber")
s.box(642, 116, 90, 68, "Top 5", ["to prompt"], "output", size=13.5)
s.arrow([(732, 150), (760, 150)], "output")
s.box(762, 116, 90, 68, "LLM", ["answers"], "model", size=14)

# candidate counts, drawn
s.grid(193, 214, 10, 10, lambda i, j: True, "compute")
s.text(243, 332, "100 candidates", size=12, weight=700, fill="#1B418C")
s.text(243, 350, "(50–200 typical)", size=11, fill="#667085")
s.text(485, 232, "one forward pass per candidate", size=11.5, fill="#7A5300", weight=600)
s.text(485, 250, "~100 on a GPU: tens to low hundreds of ms", size=11, fill="#667085")
col, fill, _ = PALETTE["output"]
for k in range(5):
    s.add(f'<rect x="{655 + k * 13}" y="214" width="11" height="11" rx="3" fill="{col}" fill-opacity="0.85"/>')
s.text(687, 244, "5 kept (3–10 typical)", size=12, weight=700, fill="#275C1C")
s.text(687, 262, "fewer tokens,", size=11, fill="#667085")
s.text(687, 278, "less lost-in-the-middle", size=11, fill="#667085")

s.add('<line x1="360" y1="300" x2="852" y2="300" stroke="#E4E7EC" stroke-width="1.5"/>')
s.text(372, 324, "It cannot fix recall: a passage missed in the first stage never reaches the reranker.", size=11.5, fill="#8E2A23", weight=600, anchor="start")
s.text(372, 344, "Pick the candidate count where first-stage recall@N plateaus.", size=11.5, fill="#344054", anchor="start")
s.text(372, 364, "A calibrated top-score threshold lets the system answer \"not found\".", size=11.5, fill="#344054", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/03-retrieval-augmented-generation-rag/q10-reranking.svg")
