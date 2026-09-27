"""LLM Fundamentals Q61: detect unanswerable questions with gates before and after generation."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(880, 440, "Unanswerable questions: check before and after generating", "Gate on retrieval, give the generator a legal no-answer, then verify every claim against its cited passage.")
Y = 136
s.pill(74, Y, 104, 40, "Question", "human")
s.arrow([(126, Y), (146, Y)], "human")
s.box(148, Y - 32, 132, 64, "Retrieve", ["+ rerank"], "compute")
s.arrow([(280, Y), (298, Y)], "compute")
s.hexagon(378, Y, 160, 56, "Score ≥ threshold?", "amber", size=12.5)
s.text(378, 82, "best reranker score vs", size=11, fill="#7A5300")
s.text(378, 97, "a calibrated threshold", size=11, fill="#7A5300")
s.arrow([(458, Y), (478, Y)], "amber")
s.hexagon(554, Y, 150, 56, "Answer present?", "amber", size=12.5)
s.text(554, 82, "relevant ≠ answerable:", size=11, fill="#7A5300")
s.text(554, 97, "classifier or judge", size=11, fill="#7A5300")
s.arrow([(629, Y), (662, Y)], "amber", label="yes", label_dy=-11)
s.box(664, Y - 36, 192, 72, "Generate", ["NO_ANSWER option + one", "example; citations required"], "model", size=13.5)
s.arrow([(760, Y + 36), (760, 238)], "model")
s.hexagon(760, 270, 150, 58, "Claims entailed?", "amber", size=12.5)
s.arrow([(760, 299), (760, 348)], "output")
s.text(772, 322, "yes", size=11.5, weight=600, fill="#275C1C", anchor="start")
s.pill(760, 370, 180, 40, "Answer + citations", "output", size=12.5)

# the no-answer sink
s.box(190, 222, 210, 78, "NO_ANSWER", ["abstain instead of guessing"], "fail", size=14)
s.arrow([(378, Y + 28), (378, 220)], "fail")
s.text(390, 196, "no", size=11.5, weight=700, fill="#8E2A23", anchor="start")
s.arrow([(554, Y + 28), (554, 246), (402, 246)], "fail")
s.text(566, 196, "no", size=11.5, weight=700, fill="#8E2A23", anchor="start")
s.arrow([(685, 270), (402, 270)], "fail")
s.text(544, 258, "no", size=11.5, weight=700, fill="#8E2A23")
s.text(544, 288, "split into claims, check each vs its passage", size=11, fill="#7A5300")

s.box(44, 336, 560, 70, "Pitfalls", ["per-passage checks miss answers that need two passages: check the joined set",
                                      "tune thresholds on an eval set that includes near-miss unanswerables"], "pink", size=13)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/01-llm-fundamentals/q61-unanswerable.svg")
