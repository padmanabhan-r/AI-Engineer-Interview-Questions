"""System design Q17: content moderation as a tiered funnel; each tier passes on only what it cannot settle."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(880, 580, "Content moderation as a tiered funnel", "Cheap, precise tiers settle most items; the LLM and then humans see only the ambiguous, severe or high-reach rest.")
cx = 310
s.pill(cx, 104, 280, 40, "Content · 100M items/day", "human")
s.arrow([(cx, 124), (cx, 150)], "human")
s.hexagon(cx, 180, 420, 56, "Hash match · PhotoDNA, PDQ", "amber")
s.arrow([(cx, 208), (cx, 240)], "slate")
s.text(cx - 10, 224, "no match", size=11.5, fill="#344054", weight=600, anchor="end")
s.box(cx - 180, 242, 360, 62, "Fast classifiers", ["every item, in milliseconds · ~3,000/s peak"], "compute", size=13.5)
s.arrow([(cx, 304), (cx, 336)], "compute")
s.text(cx - 10, 320, "uncertain or high reach", size=11.5, fill="#1B418C", weight=600, anchor="end")
s.box(cx - 155, 338, 310, 62, "LLM applies the policy text", ["~5% of items · ~150 calls/s"], "model", size=13.5)
s.arrow([(cx, 400), (cx, 432)], "model")
s.text(cx - 10, 416, "uncertain or severe", size=11.5, fill="#3B2596", weight=600, anchor="end")
s.box(cx - 130, 434, 260, 62, "Human review", ["by severity · ~0.1%, 100k/day"], "human", size=13.5)

# every tier can settle an item
s.box(640, 150, 210, 346, "Publish or enforce", ["thresholds per category,", "set by the cost", "of each error"], "output", size=14)
for y, x0, lab, role in [(180, 520, "known match", "amber"), (273, 490, "clear", "compute"),
                        (369, 465, "confident", "model"), (465, 440, "decision", "human")]:
    s.arrow([(x0, y), (638, y)], role, label=lab, label_dy=-10)
# appeals loop
s.arrow([(745, 496), (745, 540), (cx, 540), (cx, 498)], "fail")
s.text(528, 528, "appeals · overturn rate tracked", size=11.5, fill="#8E2A23", weight=600)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q17-content-moderation.svg")
