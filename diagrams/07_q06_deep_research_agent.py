"""System design Q6: a deep research agent, orchestrator-worker with a reflect loop and a writer that reads notes only."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 500, "A deep research agent", "Plan, fan out isolated workers, reflect on gaps, and let the writer cite only what the workers actually fetched.")

s.pill(84, 270, 116, 42, "Question", "human")
s.arrow([(142, 270), (170, 270)], "human")
s.box(172, 232, 136, 76, "Planner", ["sub-questions", "strong model"], "model", size=14)

# workers fan out
s.region(340, 104, 200, 334, "Workers · parallel", "compute", label_pos="bl")
for i, y in enumerate((126, 216, 306)):
    s.box(356, y, 168, 70, "Worker", ["search + read", "cheap model"], "compute", size=13.5)
    s.arrow([(308, 270), (330, 270), (330, y + 35), (354, y + 35)], "model")
    s.arrow([(524, y + 35), (548, y + 35), (548, 270), (560, 270)], "compute")
s.text(440, 460, "fresh context, one sub-question each", size=11.5, fill="#1B418C", weight=600)

s.cylinder(624, 270, 124, 96, "Notes", "data")
s.text(624, 336, "claim · quote", size=11.5, fill="#0A5A51", weight=600)
s.text(624, 353, "URL · date", size=11.5, fill="#0A5A51", weight=600)
s.arrow([(686, 270), (712, 270)], "data")
s.diamond(786, 270, 144, 104, "Reflect", "amber", size=13.5)

# gaps loop back to the planner, over the workers
s.arrow([(786, 218), (786, 88), (240, 88), (240, 230)], "amber", dashed=True)
s.text(510, 80, "gaps or conflicts: plan more sub-questions", size=11.5, fill="#7A5300", weight=600)

# done: write
s.arrow([(786, 322), (786, 398)], "output")
s.text(776, 360, "covered, or", size=11.5, fill="#275C1C", weight=600, anchor="end")
s.text(776, 377, "budget spent", size=11.5, fill="#275C1C", weight=600, anchor="end")
s.box(640, 400, 220, 70, "Writer + verifier", ["reads notes only", "every claim → a quote"], "output", size=13.5)
s.text(40, 360, "Stopping rule", size=12.5, fill="#344054", weight=700, anchor="start")
s.text(40, 382, "two independent sources", size=11.5, fill="#344054", anchor="start")
s.text(40, 399, "per sub-question,", size=11.5, fill="#344054", anchor="start")
s.text(40, 416, "or the budget is spent", size=11.5, fill="#344054", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q06-deep-research-agent.svg")
