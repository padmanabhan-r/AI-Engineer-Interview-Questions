"""System design Q41: notifications scored per user and spent from a daily budget; critical alerts bypass scoring."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 540, "Prioritised notifications", "Every candidate competes for a per-user budget: send, fold into a digest, or drop. Critical alerts skip the queue.")

# row A: critical bypass
s.box(36, 90, 170, 58, "Candidates", ["from product teams"], "slate", size=13.5)
s.arrow([(206, 119), (238, 119)], "slate")
s.diamond(306, 119, 136, 80, "Critical?", "amber")
s.arrow([(374, 119), (482, 119)], "fail", label="yes", label_dy=-11)
s.text(428, 140, "security, payments", size=11, fill="#8E2A23")
s.pill(560, 119, 150, 42, "Send now", "output")
s.arrow([(635, 119), (712, 119)], "output")
s.cylinder(790, 119, 150, 70, "Opens, dismissals,\ndisables", "data", size=12)
s.text(790, 172, "train the scorer", size=11, fill="#0A5A51", weight=600)

# row B: scoring and policy
s.arrow([(306, 159), (306, 236)], "slate")
s.text(316, 200, "no", size=11.5, fill="#344054", weight=600, anchor="start")
s.box(206, 238, 200, 66, "Score (GBDT)", ["p(open), p(disable),", "importance"], "model", size=13.5)
s.cylinder(106, 271, 136, 70, "User features", "data", size=12.5)
s.text(106, 322, "history, fatigue,\nquiet hours", size=11, fill="#0A5A51")
s.arrow([(174, 271), (204, 271)], "data")
s.box(206, 358, 200, 56, "LLM content tags", ["once per item, reused"], "model", size=13)
s.arrow([(306, 358), (306, 306)], "model")
s.arrow([(406, 271), (456, 271)], "model")
s.hexagon(560, 271, 206, 64, "Per-user budget", "amber", size=13)
s.arrow([(560, 239), (560, 142)], "output")
s.text(570, 190, "send", size=11.5, fill="#275C1C", weight=600, anchor="start")
s.arrow([(663, 271), (690, 271)], "model")
s.box(692, 240, 178, 62, "Digest builder", ["LLM summarises what", "was held back"], "model", size=13)
s.arrow([(560, 303), (560, 358)], "slate")
s.text(570, 332, "drop", size=11.5, fill="#344054", weight=600, anchor="start")
s.pill(560, 380, 130, 40, "Not sent", "slate")

# objective
s.region(24, 440, 852, 82, "", "amber", dashed=False)
s.text(450, 466, "value = p(open) · importance − λ · p(disable or uninstall)", size=14, weight=700, fill="#7A5300", mono=True)
s.text(450, 496, "The negative term is what stops spam; importance is set centrally or learned, never self-declared by teams.", size=12, fill="#344054")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q41-notification-prioritizer.svg")
