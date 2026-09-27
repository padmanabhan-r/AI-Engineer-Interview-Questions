"""Agents Q2: agent memory as a write path, a read path and consolidation around a store outside the model."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(880, 400, "Agent memory: write, read, consolidate", "The model is stateless; memory is what the harness stores outside it and puts back into the context window.")

s.box(36, 180, 170, 70, "Agent turn", ["stateless model"], "model")

# write path (top)
s.arrow([(121, 180), (121, 130), (284, 130)], "amber")
s.box(286, 100, 224, 60, "Write path", ["an LLM call extracts durable facts"], "amber", size=13.5)
s.text(398, 176, "untrusted source? store provenance,", size=11, fill="#8E2A23", weight=600)
s.text(398, 191, "or the injection persists", size=11, fill="#8E2A23", weight=600)
s.arrow([(510, 130), (640, 130), (640, 156)], "amber")

# the store
s.cylinder(640, 215, 170, 112, "Long-term store", "data")
s.text(640, 243, "vectors · profile · files", size=10.5, fill="#0A5A51")
s.text(640, 258, "(CLAUDE.md)", size=10.5, fill="#0A5A51", mono=True)

# consolidation
s.arrow([(725, 198), (752, 198)], "pink")
s.box(754, 162, 104, 106, "Consolidate", ["merge dupes,", "replace", "contradicted,", "expire"], "pink", size=12.5, detail=11)
s.arrow([(752, 238), (727, 238)], "pink")

# read path (bottom)
s.arrow([(640, 271), (640, 310), (512, 310)], "compute")
s.box(286, 280, 224, 60, "Read path", ["similarity, recency, importance"], "compute", size=13.5)
s.text(398, 356, "or the agent calls search_memory", size=11, fill="#1B418C", mono=True)
s.arrow([(286, 310), (121, 310), (121, 252)], "compute")
s.text(200, 298, "inject into context", size=11.5, fill="#1B418C", weight=600)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/04-ai-agents-and-agentic-systems/q02-agent-memory.svg")
