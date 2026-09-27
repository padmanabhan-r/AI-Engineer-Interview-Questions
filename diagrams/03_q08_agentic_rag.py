"""RAG Q8: agentic RAG, retrieval as a tool the agent calls in a loop until the results suffice."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(860, 380, "Agentic RAG: retrieval inside an agent loop", "The agent routes, rewrites and retries until a grader says the results suffice; cap the loop, it costs calls.")
s.pill(92, 170, 112, 40, "Question", "human")
s.arrow([(148, 170), (166, 170)], "human")
s.box(168, 132, 164, 76, "Agent", ["plan, route,", "rewrite the query"], "model")
s.arrow([(332, 170), (353, 170)], "model")
s.diamond(420, 170, 130, 96, "Which\ntool?", "amber")
s.arrow([(485, 170), (497, 170), (497, 118), (513, 118)], "amber")
s.arrow([(497, 170), (497, 224), (513, 224)], "amber")
s.box(515, 92, 140, 52, "Vector search", ["query, filters"], "data", size=13)
s.box(515, 198, 140, 52, "SQL over tables", ["typed arguments"], "data", size=13)
s.arrow([(655, 118), (676, 118), (676, 170), (697, 170)], "data")
s.arrow([(655, 224), (676, 224), (676, 170)], "data", head=False)
s.hexagon(775, 170, 150, 70, "Relevant and\nsufficient?", "amber", size=12.5)
# yes: answer
s.arrow([(805, 205), (805, 296)], "output")
s.text(815, 250, "yes", size=11.5, fill="#275C1C", weight=600, anchor="start")
s.pill(720, 318, 250, 42, "Grounded answer + citations", "output", size=12.5)
# no: loop back to the agent
s.arrow([(745, 205), (745, 276), (250, 276), (250, 210)], "fail", dashed=True)
s.text(498, 264, "no: rewrite, re-route or fall back (CRAG), then retry", size=11.5, fill="#8E2A23", weight=600)
s.text(498, 292, "cap iterations and tool calls", size=11, fill="#667085", italic=True)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/03-retrieval-augmented-generation-rag/q08-agentic-rag.svg")
