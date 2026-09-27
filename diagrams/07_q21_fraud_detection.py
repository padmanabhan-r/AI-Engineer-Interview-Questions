"""System design Q21: fraud detection: a tabular model decides in real time; LLMs work around that core."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 480, "Fraud detection with LLMs around the core", "Gradient-boosted trees make the real-time call; LLMs add text features, summarise cases and draft narratives.")
# real-time core
s.region(24, 80, 852, 140, "REAL-TIME · ~50–100 ms per decision, no LLM call", "compute")
s.pill(88, 160, 124, 40, "Transaction", "human", size=12)
s.arrow([(150, 160), (164, 160)], "compute")
s.box(166, 124, 190, 72, "Real-time features", ["velocity, amount vs history,", "device, shared-device graph"], "data", size=13.5)
s.arrow([(356, 160), (382, 160)], "compute")
s.box(384, 128, 140, 64, "GBDT score", ["SHAP reason codes"], "compute", size=13.5)
s.arrow([(524, 160), (548, 160)], "amber")
s.hexagon(612, 160, 128, 60, "Rules\nengine", "amber")
s.arrow([(676, 160), (702, 160)], "amber")
s.diamond(786, 160, 164, 100, "approve,\nstep-up,\ndecline", "output", size=12)

# LLMs around the core
s.region(24, 238, 852, 222, "LLMs AROUND THE CORE · offline and in the analyst workflow", "model", label_pos="br")
s.arrow([(786, 210), (786, 262)], "output")
s.cylinder(786, 296, 130, 64, "Case queue", "slate", size=12.5)
s.arrow([(721, 296), (672, 296)], "slate")
s.box(460, 264, 210, 64, "LLM copilot", ["case summary, draft narrative", "for sign-off"], "model", size=13.5)
s.arrow([(460, 296), (412, 296)], "model")
s.pill(360, 296, 100, 40, "Analyst", "human")
s.arrow([(360, 276), (360, 256), (454, 256), (454, 194)], "human", dashed=True)
s.text(462, 244, "labels for retraining, weeks later", size=11, fill="#1F3864", weight=600, anchor="start")
# offline text features
s.box(40, 374, 146, 60, "Dispute text", ["merchant descriptors"], "slate", size=13)
s.arrow([(186, 404), (206, 404)], "model")
s.box(208, 374, 190, 60, "LLM offline features", ["from unstructured text"], "model", size=13)
s.arrow([(260, 374), (260, 198)], "model")
s.text(762, 206, "cost-based threshold", size=11, fill="#275C1C", weight=600, anchor="end")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q21-fraud-detection.svg")
