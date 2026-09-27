"""Prompt Engineering Q5: tree-of-thought as search, with the three parts that run it."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 440, "Tree-of-thought: reasoning as search", "The model proposes and rates partial solutions; search code keeps the promising ones and backtracks from dead ends.")

# the tree
s.pill(290, 108, 150, 38, "Problem", "human")
xs = {"A": 120, "B": 290, "C": 460}
for k, x in xs.items():
    s.arrow([(290, 127), (x, 186)], "model")
s.text(386, 150, "propose k thoughts", size=11, fill="#3B2596", weight=600, anchor="start")
s.box(xs["A"] - 75, 188, 150, 58, "Thought A", ["rated: likely"], "amber", size=13.5)
s.box(xs["B"] - 75, 188, 150, 58, "Thought B", ["rated: impossible"], "fail", size=13.5)
s.box(xs["C"] - 75, 188, 150, 58, "Thought C", ["rated: sure"], "output", size=13.5)

# level 2
s.arrow([(xs["A"], 246), (xs["A"], 306)], "amber")
s.text(xs["A"] + 8, 276, "expand", size=11, fill="#7A5300", weight=600, anchor="start")
s.box(xs["A"] - 75, 308, 150, 58, "A1", ["dead end"], "fail", size=13.5)
s.arrow([(xs["A"] - 75, 337), (18 + 12, 337), (18 + 12, 217), (xs["A"] - 77, 217)], "fail", dashed=True, head=True, curve=False)
s.text(38, 277, "backtrack", size=11, fill="#8E2A23", weight=600, anchor="start")

s.arrow([(xs["B"], 246), (xs["B"], 312)], "fail", dashed=True)
s.pill(xs["B"], 332, 120, 36, "Pruned", "slate", size=12.5)

s.arrow([(xs["C"], 246), (xs["C"], 306)], "output")
s.text(xs["C"] + 8, 276, "expand", size=11, fill="#275C1C", weight=600, anchor="start")
s.box(xs["C"] - 75, 308, 150, 58, "C1", ["solved"], "output", size=13.5)

s.text(290, 404, "tens to hundreds of calls per problem", size=11.5, fill="#667085", italic=True)

# the three parts
s.region(574, 84, 282, 330, "What runs the search", "slate", dashed=False)
rows = [("model", "Generator prompt", ["propose k next thoughts", "from the current state"]),
        ("amber", "Evaluator prompt", ["rate each state:", "sure / likely / impossible"]),
        ("compute", "Search code (BFS / DFS)", ["keep the best b per level,", "prune the rest"])]
for i, (role, t, lines) in enumerate(rows):
    s.box(592, 120 + i * 96, 246, 80, t, lines, role, size=13.5)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/02-prompt-engineering/q05-tree-of-thought.svg")
