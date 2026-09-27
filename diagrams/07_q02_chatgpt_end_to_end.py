"""System design Q2: ChatGPT end to end, a training pipeline and a serving system joined by a checkpoint."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 500, "ChatGPT: training to serving", "Two systems joined by a checkpoint, with consented feedback flowing from serving back into post-training.")

# lanes first, so arrows crossing between them sit on top
s.region(24, 318, 852, 162, "SERVING", "compute", label_pos="bl")
# training lane
s.region(24, 80, 852, 136, "TRAINING", "data")
s.box(44, 112, 172, 84, "Data", ["crawl, dedup, filter,", "decontaminate"], "data", size=14)
s.arrow([(216, 154), (244, 154)], "data")
s.box(246, 112, 172, 84, "Pretraining", ["data + tensor +", "pipeline parallel"], "compute", size=14)
s.arrow([(418, 154), (446, 154)], "compute")
s.box(448, 112, 172, 84, "Post-training", ["SFT, then", "RLHF or DPO"], "model", size=14)
s.arrow([(620, 154), (660, 154)], "model")
s.hexagon(768, 154, 200, 70, "Capability +\nsafety evals", "amber", size=13)

# checkpoint crosses to serving
s.arrow([(768, 189), (768, 336)], "amber", width=2.4)
s.text(780, 262, "checkpoint", size=12, fill="#7A5300", weight=700, anchor="start")
s.text(780, 280, "canary first", size=11.5, fill="#7A5300", anchor="start")

# feedback store sits between the lanes
s.cylinder(534, 276, 150, 66, "Feedback logs", "slate", size=12.5)
s.arrow([(534, 243), (534, 198)], "slate", dashed=True)
s.text(526, 226, "consented", size=11.5, fill="#344054", weight=600, anchor="end")

# serving lane
s.pill(100, 390, 120, 42, "Users", "human")
s.arrow([(160, 390), (190, 390)], "human")
s.hexagon(282, 390, 180, 62, "Gateway +\nmoderation", "amber", size=12.5)
s.arrow([(372, 390), (446, 390)], "compute")
s.box(448, 350, 172, 80, "Orchestrator", ["history, tools"], "compute", size=14)
s.arrow([(534, 350), (534, 311)], "slate")
s.arrow([(620, 390), (656, 390)], "compute")
s.box(658, 338, 200, 104, "Inference fleet", ["continuous batching", "paged KV cache", "prefix caching"], "model", size=14)
s.text(758, 462, "KV memory caps concurrency", size=11.5, fill="#3B2596", weight=600)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q02-chatgpt-end-to-end.svg")
