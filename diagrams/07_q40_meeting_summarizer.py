"""System design Q40: a meeting summariser as a batch pipeline, with action items as verified objects."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 584, "An AI meeting summariser", "A batch pipeline: consent, ASR, named speakers, one structured LLM call, then every item checked against the transcript.")

s.region(24, 78, 852, 226, "BATCH · 10k meetings/day, results within ~10 min", "compute")
# row 1, left to right
s.box(40, 116, 160, 60, "Meetings", ["Zoom, Teams, Meet"], "human", size=13.5)
s.arrow([(200, 146), (222, 146)], "amber")
s.hexagon(298, 146, 148, 54, "Consent check", "amber", size=12.5)
s.arrow([(372, 146), (394, 146)], "data")
s.cylinder(460, 146, 128, 62, "Job queue", "data")
s.arrow([(524, 146), (552, 146)], "compute")
s.box(554, 116, 200, 60, "ASR + diarisation", ["custom vocabulary from invite"], "compute", size=13.5)
s.text(460, 192, "bursty at :00 and :30", size=11, fill="#0A5A51", weight=600)

# row 2, right to left
s.arrow([(654, 176), (654, 220)], "compute")
s.box(554, 222, 200, 60, "Speaker → attendee", ["from platform metadata"], "compute", size=13.5)
s.arrow([(554, 252), (522, 252)], "model")
s.box(300, 222, 220, 60, "LLM, one call", ["summary, decisions, actions"], "model", size=14)
s.arrow([(300, 252), (272, 252)], "amber")
s.hexagon(186, 252, 170, 56, "Verify quotes\n+ timestamps", "amber", size=12.5)

# row 3: outputs
s.arrow([(186, 280), (186, 322)], "output")
s.box(86, 324, 200, 58, "Deliver", ["email, chat, CRM, tasks"], "output", size=13.5)
s.arrow([(700, 282), (700, 322)], "data")
s.cylinder(700, 353, 210, 62, "Transcript search", "data", size=12.5)
s.text(700, 396, "attendees only", size=11, fill="#0A5A51", weight=600)
s.text(470, 342, "Templates by meeting type:", size=11.5, fill="#3B2596", weight=700)
s.text(470, 362, "sales, one-to-one, stand-up", size=11.5, fill="#3B2596")

# action item as an object
s.region(24, 418, 852, 148, "An action item is an object, not a sentence", "slate", dashed=False)
fields = [("owner", "named attendee, or \"unassigned\""), ("task", "what was committed to"), ("due", "date, if one was said"),
          ("timestamp", "where it was said"), ("quote", "the words that prove it")]
for i, (k, v) in enumerate(fields):
    y = 456 + i * 21
    s.text(48, y, k, size=11.5, fill="#1B418C", weight=700, anchor="start", mono=True)
    s.text(140, y, v, size=11.5, fill="#344054", anchor="start")
col, fill, ink = PALETTE["fail"]
s.add(f'<rect x="448" y="450" width="408" height="98" rx="10" fill="{fill}" stroke="{col}" stroke-opacity="0.5"/>')
s.text(464, 474, "“Priya might send the deck”", size=12.5, fill=ink, weight=700, anchor="start")
s.text(464, 498, "→ a suggestion, not a commitment", size=12, fill=ink, anchor="start")
s.text(464, 526, "No explicit owner → \"unassigned\", never a guess", size=12, fill=ink, anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q40-meeting-summarizer.svg")
