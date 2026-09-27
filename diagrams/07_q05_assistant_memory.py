"""System design Q5: memory for a personal assistant, a fast read path in the request and an async write path after it."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 500, "Memory for a personal assistant", "The model remembers nothing; a read path injects a few memories, and an async write path keeps the store current.")

s.region(24, 80, 852, 170, "READ PATH · in the request, < ~150 ms added", "compute")
s.region(24, 276, 852, 204, "WRITE PATH · async, after the reply", "slate", label_pos="bl")

# read path, left to right
s.pill(100, 150, 128, 42, "User turn", "human")
s.arrow([(164, 150), (194, 150)], "human")
s.box(196, 112, 240, 76, "Retrieve", ["similarity + recency + importance", "capped at ~500 tokens"], "compute", size=14)
s.arrow([(436, 150), (476, 150)], "compute")
s.box(478, 118, 180, 64, "Assistant LLM", ["memories in the prompt"], "model", size=14)
s.arrow([(658, 150), (696, 150)], "model")
s.pill(768, 150, 136, 42, "Reply", "output")
s.text(250, 216, "read score  s = α·sim + β·e^(−λΔt) + γ·importance", size=12, fill="#1B418C", mono=True, anchor="start")

# write path, right to left
s.arrow([(768, 171), (768, 318)], "slate")
s.pill(768, 340, 136, 42, "Async queue", "slate", size=12.5)
s.arrow([(700, 340), (660, 340)], "slate")
s.box(478, 304, 180, 72, "Extract", ["candidate memories,", "user's own messages only"], "amber", size=14)
s.arrow([(478, 340), (438, 340)], "amber")
s.box(256, 304, 180, 72, "Reconcile", ["add · update", "merge · ignore"], "amber", size=14)
s.arrow([(256, 340), (212, 340)], "data")
s.cylinder(126, 340, 168, 84, "Per-user store", "data", size=12.5)
s.text(126, 404, "timestamps + provenance", size=11.5, fill="#0A5A51", weight=600)
s.text(126, 422, "view · edit · delete", size=11.5, fill="#0A5A51")
s.arrow([(126, 298), (126, 262), (226, 262), (226, 190)], "data", dashed=True)

# memory types
s.text(270, 436, "types: semantic (facts)  ·  episodic (past sessions)  ·  procedural (how they like it)", size=11.5, fill="#344054", anchor="start")
s.text(270, 402, "update: \"moved to Berlin\" supersedes London", size=11.5, fill="#7A5300", weight=600, anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q05-assistant-memory.svg")
