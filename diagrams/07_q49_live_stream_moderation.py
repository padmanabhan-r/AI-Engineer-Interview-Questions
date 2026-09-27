"""System design Q49: live-stream moderation: risk-based sampling, fast classifiers escalating to a VLM, a stream risk score, graduated actions."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 560, "Moderating live streams", "Sample by risk, score fast, escalate the uncertain, and act within seconds; a pattern cuts a stream, one frame does not.")

s.region(24, 78, 604, 300, "SIGNALS · 100k concurrent streams", "compute")
# chat lane
s.pill(96, 134, 112, 40, "Chat", "slate")
s.arrow([(150, 134), (164, 134)], "compute")
s.box(166, 108, 190, 52, "Chat classifier", [], "compute", size=13.5)
s.arrow([(356, 134), (752, 134), (752, 170)], "compute")
# video / audio lane
s.pill(97, 228, 114, 40, "Live stream", "human", size=11.5)
s.arrow([(150, 228), (164, 228)], "compute")
s.box(166, 186, 190, 84, "Risk-based sampler", ["frames + audio; denser for", "risky creators, big reach;", "random + scene-change"], "compute", size=13)
s.arrow([(356, 228), (392, 228)], "compute")
s.box(394, 196, 190, 64, "Fast classifiers", ["vision + audio"], "compute", size=13.5)
s.arrow([(584, 228), (650, 228)], "compute")
s.arrow([(489, 260), (489, 296)], "model")
s.text(499, 278, "uncertain or high reach", size=11, fill="#3B2596", weight=600, anchor="start")
s.box(394, 298, 190, 60, "VLM", ["applies policy with context"], "model", size=13.5)
s.arrow([(584, 328), (650, 328)], "model")
s.text(40, 316, "Broadcast delay of a few", size=11.5, fill="#1F3864", weight=600, anchor="start")
s.text(40, 334, "seconds on high-risk", size=11.5, fill="#1F3864", weight=600, anchor="start")
s.text(40, 352, "categories buys time to act.", size=11.5, fill="#1F3864", weight=600, anchor="start")

# risk score
s.box(652, 172, 210, 190, "Stream risk score", ["sliding window:", "one ambiguous frame", "does not cut a stream,", "a pattern does"], "amber", size=14)

# actions and people
s.arrow([(800, 362), (800, 424)], "human")
s.box(692, 426, 170, 62, "Human moderators", ["by priority"], "human", size=13)
s.arrow([(672, 362), (672, 452), (606, 452)], "fail")
s.arrow([(692, 474), (606, 474)], "human")
s.region(24, 398, 580, 140, "ACTIONS · graduated", "fail", dashed=False)
for cx, label, role in ((96, "warn", "amber"), (226, "blur", "amber"), (362, "cut stream", "fail"), (508, "suspend", "fail")):
    s.pill(cx, 460, 116 if label != "cut stream" else 124, 38, label, role, size=12.5)
s.text(226, 494, "auto if high-confidence", size=11, fill="#7A5300", weight=600)
s.text(226, 510, "and severe", size=11, fill="#7A5300", weight=600)
s.text(508, 494, "ban only with", size=11, fill="#8E2A23", weight=600)
s.text(508, 510, "human review", size=11, fill="#8E2A23", weight=600)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q49-live-stream-moderation.svg")
