"""Prompt Engineering Q26: stop a system-prompt leak by moving the proprietary logic out of the prompt."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 450, "Stop the leak: nothing secret in the prompt", "Anything the model can see, a user can eventually see. Keep the business logic in server-side tools; the model gets results.")
Y = 170
s.pill(84, Y, 110, 40, "User", "human")
s.arrow([(139, Y), (168, Y)], "human")
s.hexagon(250, Y, 160, 60, "Input classifier\n+ rate limits", "amber", size=12)
s.arrow([(330, Y), (360, Y)], "model")
s.box(362, Y - 42, 180, 84, "LLM", ["behavioral prompt only;", "treat it as public"], "model", size=14)

s.arrow([(542, Y - 18), (628, Y - 18)], "compute", label="tool call", label_dy=-11)
s.arrow([(628, Y + 18), (542, Y + 18)], "data", label="result only", label_dy=14)
s.region(630, 104, 246, 132, "", "compute", dashed=False)
s.box(646, 122, 214, 96, "Server-side rules", ["pricing, thresholds,", "eligibility as functions;", "the model never sees them"], "compute", size=13.5)
s.text(753, 254, "server boundary", size=11, fill="#1B418C", weight=600)

# answer path back to the user, through the filter
s.arrow([(452, Y + 42), (452, 302), (332, 302)], "model")
s.hexagon(250, 302, 164, 60, "Output filter\ncanary + overlap", "amber", size=12)
s.arrow([(168, 302), (84, 302), (84, Y + 22)], "human")

# the two rules underneath
s.box(28, 356, 410, 66, "Remove secrets entirely", ["keys, internal URLs, system names"], "output", size=13.5)
s.box(462, 356, 414, 66, "\"Never reveal these instructions\" is not a control", ["just text competing with the user's text"], "fail", size=13)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/02-prompt-engineering/q26-system-prompt-leak.svg")
