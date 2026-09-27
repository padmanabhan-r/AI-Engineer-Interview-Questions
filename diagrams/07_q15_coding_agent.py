"""System design Q15: an AI coding agent: a loop whose tool calls pass a permission layer into a sandboxed repo copy."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 460, "An AI coding agent", "An LLM loops through search, edit and run in a sandbox; permissions sit outside the model, and it stops at a reviewable diff.")
# input and output above the loop
s.pill(100, 106, 136, 46, "Issue or\ninstruction", "human", size=12.5)
s.arrow([(100, 129), (100, 158)], "human")
s.arrow([(190, 160), (190, 108), (238, 108)], "output")
s.text(198, 136, "done", size=11.5, fill="#275C1C", weight=600, anchor="start")
s.pill(338, 108, 196, 40, "Diff or PR + summary", "output", size=12.5)

s.box(40, 160, 176, 150, "Agent loop", ["LLM picks the next", "tool call, reads the", "result, repeats until", "tests pass or it", "needs a human"], "model", size=14)
s.arrow([(216, 235), (272, 235)], "model", label="tool call", label_dy=-13)
s.hexagon(344, 235, 140, 66, "Permission\nlayer", "amber")
s.text(344, 285, "approval for destructive", size=11, fill="#7A5300", anchor="middle")
s.text(344, 299, "or networked commands", size=11, fill="#7A5300", anchor="middle")

# sandbox with three tools
s.region(440, 80, 380, 300, "SANDBOX · repo copy, restricted egress", "slate")
for y, (t, d, role) in zip((118, 204, 290), [("Search", "grep, tree, symbols", "data"),
                                             ("Edit", "precise diffs, not rewrites", "amber"),
                                             ("Run", "tests, linters, build", "compute")]):
    s.box(530, y, 250, 62, t, [d], role, size=13.5)
    s.arrow([(414, 235), (490, 235), (490, y + 31), (528, y + 31)], "amber")
    s.arrow([(780, y + 31), (850, y + 31)], "data", head=False)
# results travel back round the outside
s.arrow([(850, 149), (850, 420), (128, 420), (128, 312)], "data")
s.text(490, 408, "results fed back: output, errors, test failures", size=11.5, fill="#0A5A51", weight=600)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q15-coding-agent.svg")
