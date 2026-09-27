"""System design Q27: conversational memory across sessions: a durable log, consolidated at session end into scoped memory."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 500, "Conversational memory across sessions", "The raw log is the source of truth; summaries and facts are derived at session end and loaded back within scope.")
# in session
s.region(24, 80, 852, 170, "IN SESSION", "compute")
s.pill(106, 134, 144, 40, "New session", "human")
s.arrow([(178, 134), (206, 134)], "compute")
s.box(208, 108, 210, 52, "Load profile", ["+ pinned facts"], "compute", size=13.5)
s.pill(106, 206, 144, 40, "Message", "human")
s.arrow([(178, 206), (206, 206)], "compute")
s.box(208, 178, 210, 56, "Retrieve past sessions", ["relevant summaries, in scope"], "data", size=13)
s.arrow([(418, 134), (568, 134)], "compute")
s.arrow([(418, 206), (568, 206)], "data")
s.box(570, 110, 150, 120, "LLM", ["shows which past", "conversation it used"], "model", size=15)
s.arrow([(720, 170), (752, 170)], "slate")
s.cylinder(812, 170, 116, 88, "Durable\nlog", "slate", size=12.5)

# at session end
s.region(24, 268, 852, 214, "AT SESSION END · async", "data", label_pos="bl")
s.arrow([(812, 214), (812, 300)], "slate")
s.pill(806, 322, 128, 40, "Session end", "slate", size=12.5)
s.arrow([(742, 322), (722, 322)], "data")
s.box(512, 290, 208, 64, "Consolidate", ["~200-token summary,", "reconcile facts"], "data", size=13.5)
s.arrow([(512, 322), (452, 322)], "data")
s.cylinder(313, 334, 276, 88, "Scoped memory:\nsummaries + facts", "data", size=13)
s.arrow([(313, 290), (313, 236)], "data")
# scope chips
for x, lab, role in [(215, "global", "output"), (313, "workspace", "compute"), (411, "session-only", "amber")]:
    st, f, ink = PALETTE[role]
    s.add(f'<rect x="{x - 44}" y="{396}" width="88" height="24" rx="12" fill="{f}" stroke="{st}" stroke-width="1.3"/>')
    s.text(x, 408, lab, size=11, weight=600, fill=ink)
s.text(313, 438, "temporary chats are never written here", size=11, fill="#8E2A23", weight=600)
s.text(500, 400, "deletion cascades via provenance", size=11, fill="#0A5A51", weight=600, anchor="start")
s.text(500, 416, "links to summaries, facts and caches", size=11, fill="#0A5A51", weight=600, anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q27-cross-session-memory.svg")
