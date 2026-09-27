"""Agents Q36: a support agent as a graph with a bounded agent inside and code-enforced escalation."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 600, "Support agent with escalation", "A bounded agent inside a graph; code enforces the escalation triggers. Optimize for correct escalation, not maximum containment.")
X = 300  # the spine

s.pill(X, 104, 180, 38, "Customer message", "human", size=12.5)
s.arrow([(X, 123), (X, 146)], "human")
s.hexagon(X, 170, 180, 46, "Verify identity", "amber", size=12.5)
s.text(X + 102, 170, "then scope every account tool to that ID", size=11, fill="#7A5300", anchor="start")
s.arrow([(X, 193), (X, 214)], "slate")
s.diamond(X, 250, 170, 72, "Triage", "slate")
s.text(X - 92, 250, "intent · risk · sentiment", size=11, fill="#344054", anchor="end")

# escalate straight away
s.arrow([(X + 85, 250), (618, 250)], "fail")
s.text(502, 238, "high risk, or asks for a human", size=11.5, fill="#8E2A23", weight=600)

s.arrow([(X, 286), (X, 318)], "model")
s.text(X + 10, 302, "in scope", size=11.5, fill="#3B2596", weight=600, anchor="start")
s.box(X - 100, 320, 200, 60, "Agent", ["KB retrieval + scoped tools"], "model")
s.arrow([(X, 380), (X, 408)], "slate")
s.diamond(X, 444, 190, 70, "Action over limit?", "amber", size=12)

# approval loop
s.arrow([(X - 95, 444), (152, 444)], "human")
s.text(180, 432, "yes", size=11.5, fill="#1F3864", weight=600)
s.box(30, 414, 120, 60, "Human\napproval", (), "human", size=13)
s.arrow([(90, 414), (90, 350), (198, 350)], "human")

s.arrow([(X, 479), (X, 512)], "slate")
s.text(X + 10, 495, "no", size=11.5, fill="#344054", weight=600, anchor="start")
s.hexagon(X, 538, 180, 48, "Grounded check", "amber", size=12.5)
s.text(X, 574, "policy answers must cite passages", size=10.5, fill="#7A5300")
s.arrow([(X - 90, 538), (152, 538)], "output")
s.text(182, 526, "pass", size=11.5, fill="#275C1C", weight=600)
s.pill(90, 538, 120, 38, "Reply", "output")

# escalate after failing
s.arrow([(X + 90, 538), (745, 538), (745, 434)], "fail")
s.text(560, 526, "fails twice, or no progress", size=11.5, fill="#8E2A23", weight=600)

s.box(620, 222, 250, 210, "", (), "human")
s.text(748, 248, "Escalate with a handoff", size=13.5, weight=700, fill="#1F3864")
for i, line in enumerate(["summary of the issue", "identity status", "steps tried + tool results", "sentiment",
                          "→ ticket queue, with priority", "tell the customer what's next", "agent stops replying"]):
    s.text(646, 276 + i * 21, line, size=11.5, fill="#1F3864", anchor="start", opacity=0.9, weight=600 if line[0] == "→" else 400)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/04-ai-agents-and-agentic-systems/q36-support-agent-escalation.svg")
