"""Evaluation Q18: evaluating a RAG system: retrieval, generation and end to end, each with its own metrics."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 470, "Evaluating RAG end to end", "Score retrieval and generation separately, then the whole answer, on answerable, unanswerable and multi-hop questions.")
Y = 150

# question with its test mix
s.box(32, Y - 50, 170, 100, "Question", ["answerable", "unanswerable", "multi-hop"], "human", size=14)
s.arrow([(202, Y), (252, Y)], "human")
s.box(254, Y - 34, 160, 68, "Retriever", ["top-k chunks"], "data", size=14)
s.arrow([(414, Y), (464, Y)], "data")
s.box(466, Y - 34, 160, 68, "Generator", ["LLM over chunks"], "model", size=14)
s.arrow([(626, Y), (676, Y)], "model")
s.pill(772, Y, 188, 50, "Answer + citations", "output", size=13)

# metric cards under each stage
cards = [(254, 160, "Retrieval", ["recall@k", "MRR, nDCG", "vs labelled chunks"], "data"),
         (466, 160, "Generation", ["faithfulness", "answer relevance"], "model"),
         (678, 188, "End to end", ["correctness", "citation accuracy", "abstention", "latency, cost"], "output")]
for x, w, head, lines, role in cards:
    st, f, ink = PALETTE[role]
    cx = x + w / 2
    s.arrow([(cx, Y + (25 if role == "output" else 34)), (cx, 236)], role, dashed=True)
    s.add(f'<rect x="{x}" y="238" width="{w}" height="{32 + 18 * len(lines)}" rx="12" fill="#FFFFFF" stroke="{st}" stroke-width="1.6"/>')
    s.text(cx, 256, head, size=12.5, weight=700, fill=ink)
    for i, l in enumerate(lines):
        s.text(cx, 278 + i * 18, l, size=11.5, fill="#344054")

# diagnosis
s.text(32, 250, "Where is it broken?", size=12.5, weight=700, fill="#344054", anchor="start")
s.text(32, 274, "recall low", size=11.5, weight=600, fill="#0A5A51", anchor="start")
s.text(32, 292, "→ fix retrieval", size=11.5, fill="#344054", anchor="start")
s.text(32, 322, "recall high,", size=11.5, weight=600, fill="#3B2596", anchor="start")
s.text(32, 340, "correctness low", size=11.5, weight=600, fill="#3B2596", anchor="start")
s.text(32, 358, "→ fix the generator", size=11.5, fill="#344054", anchor="start")

s.add('<rect x="32" y="398" width="836" height="50" rx="12" fill="#FDE8E6" stroke="#D0443A" stroke-opacity="0.5"/>')
s.text(450, 423, "Pitfall: testing only answerable questions. The worst failure is a confident answer the corpus cannot support.", size=12, weight=600, fill="#8E2A23")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/09-evaluation-and-testing/q18-rag-eval.svg")
