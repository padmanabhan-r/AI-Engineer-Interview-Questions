"""System design Q19: an AI email assistant: cheap triage on every email, LLM on demand, the user confirms every action."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 470, "An AI-powered email assistant", "A cheap classifier triages every email; the LLM runs on demand and only proposes, because email is untrusted input.")
# every email: cheap tier
s.region(24, 80, 852, 130, "EVERY EMAIL · cheap tier", "data")
s.pill(98, 150, 136, 40, "Mail webhook", "human")
s.arrow([(165, 150), (196, 150)], "data")
s.box(198, 120, 170, 60, "Parse", ["strip quoted history"], "data", size=13.5)
s.arrow([(368, 150), (408, 150)], "data")
s.box(410, 116, 220, 68, "Triage classifier", ["per-user: priority, needs-reply", "biased toward surfacing"], "compute", size=13.5)
s.arrow([(630, 150), (668, 150)], "output")
s.pill(760, 150, 180, 40, "Priority inbox", "output")

# on demand: LLM tier
s.region(24, 228, 852, 222, "ON DEMAND · ~10 LLM calls per user per day", "model", label_pos="bl")
s.arrow([(283, 180), (283, 244)], "data")
s.cylinder(283, 282, 170, 70, "Per-user\nthread index", "data", size=12.5)
s.arrow([(283, 317), (283, 348)], "data")
s.pill(100, 380, 130, 40, "User asks", "human")
s.arrow([(165, 380), (196, 380)], "human")
s.box(198, 350, 170, 62, "Context", ["thread + past replies", "as style examples"], "slate", size=13.5)
s.arrow([(368, 380), (408, 380)], "model")
s.box(410, 350, 170, 62, "LLM proposes", ["a draft or an action"], "model", size=13.5)
s.arrow([(580, 380), (606, 380)], "model")
s.hexagon(680, 380, 146, 62, "User\nconfirms", "human")
s.arrow([(753, 380), (778, 380)], "output")
s.pill(826, 380, 92, 40, "Send", "output")
s.text(680, 426, "no send, delete or forward", size=11, fill="#1F3864")
s.text(680, 440, "without it", size=11, fill="#1F3864")
# injection note
s.box(420, 246, 440, 64, "Email is data, not instructions", ["no outbound tools on untrusted context;", "no auto-rendered remote images or links"], "fail", size=13)
s.arrow([(495, 311), (495, 348)], "fail", dashed=True, head=False)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q19-email-assistant.svg")
