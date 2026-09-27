"""Vector DBs Q25: search quality crashed after an embedding-model deploy: roll back, diagnose, re-roll with gates."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 500, "Search crashed after a model deploy: treat it as an incident", "An embedding model change is a data migration, not a library upgrade. Roll back first, then find the mixed spaces.")

s.box(28, 84, 390, 96, "1. Mitigate", ["flip the alias back to the last good query", "model and index; pause re-embedding"], "fail", size=13.5)
s.arrow([(418, 132), (460, 132)], "slate")
s.box(462, 84, 410, 96, "2. Diagnose", ["mixed spaces? check the embedding_model values in the index;", "then prefixes, normalization, metric, max input length,", "stale IVF centroids"], "amber", size=13.5, detail=11)
s.arrow([(667, 180), (667, 226)], "slate")

# re-roll, gated
s.region(24, 208, 852, 196, "3. Re-roll blue/green", "slate", label_pos="tl")
Y = 280
s.box(44, Y - 28, 140, 56, "Build index B", ["full re-embed"], "data", size=13.5)
s.arrow([(184, Y), (206, Y)], "data")
s.hexagon(270, Y, 124, 56, "Offline eval", "amber", size=12.5)
s.arrow([(332, Y), (362, Y)], "amber")
s.box(364, Y - 28, 140, 56, "Shadow traffic", ["not served"], "slate", size=13.5)
s.arrow([(504, Y), (538, Y)], "slate")
s.hexagon(604, Y, 124, 56, "Canary", "amber", size=12.5)
s.arrow([(666, Y), (700, Y)], "output")
s.box(702, Y - 28, 150, 56, "Atomic cutover", ["query model + index"], "output", size=13.5)

s.arrow([(270, Y + 28), (270, 348)], "fail", dashed=True)
s.text(280, 330, "worse", size=11.5, fill="#8E2A23", weight=600, anchor="start")
s.pill(270, 368, 90, 34, "Stop", "fail", size=12.5)
s.arrow([(604, Y + 28), (604, 348)], "fail", dashed=True)
s.text(614, 330, "metrics drop", size=11.5, fill="#8E2A23", weight=600, anchor="start")
s.pill(604, 368, 170, 34, "Roll back alias", "fail", size=12.5)

s.box(28, 422, 844, 58, "4. Prevent", ["fail fast on a model/index mismatch · a retrieval eval in CI · alerts on zero-result rates and score shifts"], "output", size=13.5)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/06-vector-databases-and-embeddings/q25-embedding-incident.svg")
