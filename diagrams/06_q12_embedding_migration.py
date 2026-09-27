"""Vector DBs Q12: a model update as a blue/green re-embedding migration, with its two rollback paths."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 408, "A new embedding model is a blue/green migration", "Vectors from different models live in incompatible spaces, so re-embed everything, prove it, then flip one alias.")
Y, W, G = 150, 140, 64
xs = [64 + i * (W + G) for i in range(4)]
states = [("Blue: serving", ["index A, model A"], "compute"),
          ("Backfill", ["index B from stored", "source text; new", "docs go to both"], "amber"),
          ("Shadow", ["B sees live traffic,", "results not served"], "slate"),
          ("Cutover", ["one alias flips the", "query model + index"], "output")]
for x, (t, lines, role) in zip(xs, states):
    s.box(x, Y - 44, W, 88, t, lines, role, size=13.5, detail=11)
# start and end
s.add(f'<circle cx="38" cy="{Y}" r="8" fill="#344054"/>')
s.arrow([(46, Y), (62, Y)], "slate")
s.arrow([(xs[3] + W + 2, Y), (856, Y)], "slate")
s.add(f'<circle cx="866" cy="{Y}" r="9" fill="#FFFFFF" stroke="#344054" stroke-width="2"/><circle cx="866" cy="{Y}" r="5" fill="#344054"/>')
s.text(838, Y - 26, "retire\nA", size=11, fill="#344054", weight=600)

labels = ["model B\nreleased", "index B\ncomplete", "B beats A\non eval"]
for i, lab in enumerate(labels):
    x0 = xs[i] + W
    s.arrow([(x0 + 2, Y), (x0 + G - 2, Y)], "slate")
    s.text(x0 + G / 2, Y - 26, lab, size=11, fill="#344054", weight=600)

# rollback paths
s.arrow([(xs[2] + W / 2, Y + 44), (xs[2] + W / 2, Y + 84), (xs[0] + 96, Y + 84), (xs[0] + 96, Y + 46)], "fail", dashed=True)
s.text((xs[0] + 96 + xs[2] + W / 2) / 2 + 20, Y + 72, "B worse: stay on A", size=11.5, fill="#8E2A23", weight=600)
s.arrow([(xs[3] + W / 2, Y + 44), (xs[3] + W / 2, Y + 124), (xs[0] + 44, Y + 124), (xs[0] + 44, Y + 46)], "fail", dashed=True)
s.text((xs[0] + 44 + xs[3] + W / 2) / 2 + 40, Y + 112, "regression after cutover: flip the alias back (old index kept)", size=11.5, fill="#8E2A23", weight=600)

s.box(28, 310, 410, 72, "Tag every vector", ["with model + version; name indexes by version"], "slate", size=13.5)
s.box(462, 310, 410, 72, "Same dimensions, different space", ["no error, just near-random results;", "a learned projection is a lossy stopgap"], "fail", size=13.5)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/06-vector-databases-and-embeddings/q12-embedding-migration.svg")
