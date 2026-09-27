"""Multimodal Q10: handling barge-in in a voice agent, plus the history truncation it requires."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 550, "Barge-in: stop, cancel, truncate", "When the user talks over the agent, cut playback fast and keep only what the user actually heard.")
X = {"U": 100, "C": 330, "S": 610}
for k, name, role, w in [("U", "User", "human", 100), ("C", "Client", "compute", 110), ("S", "Server", "model", 110)]:
    s.add(f'<line x1="{X[k]}" y1="126" x2="{X[k]}" y2="440" stroke="#C9CED8" stroke-width="1.5" stroke-dasharray="4 4"/>')
    s.pill(X[k], 108, w, 36, name, role, size=12.5)

def msg(a, b, y, label, role):
    d = 4 if X[b] > X[a] else -4
    s.arrow([(X[a] + d, y), (X[b] - 1.5 * d, y)], role, label=label)

msg("S", "C", 158, "streaming agent audio", "model")
msg("U", "C", 204, "starts speaking", "human")
s.text(215, 222, "min speech duration; ignore “mm-hmm”", size=10.5, fill="#667085")
msg("C", "S", 256, "speech detected after echo cancellation", "compute")
msg("S", "C", 302, "stop playback + flush buffers", "model")
msg("C", "S", 348, "played up to 2.4 s", "compute")
s.arrow([(X["S"] + 4, 372), (X["S"] + 44, 372), (X["S"] + 44, 398), (X["S"] + 8, 398)], "fail")
s.text(X["S"] + 54, 378, "cancel LLM + TTS;", size=11.5, weight=600, fill=PALETTE["fail"][2], anchor="start")
s.text(X["S"] + 54, 394, "truncate turn to heard text", size=11.5, weight=600, fill=PALETTE["fail"][2], anchor="start")
msg("S", "C", 428, "respond to the new user turn", "model")

# what gets stored as the assistant turn
s.text(36, 474, "Assistant turn stored in history", size=12, weight=700, fill="#344054", anchor="start")
O, L = PALETTE["output"], PALETTE["slate"]
s.add(f'<rect x="36" y="488" width="384" height="30" rx="7" fill="{O[1]}" stroke="{O[0]}" stroke-width="1.6"/>')
s.text(228, 503, "heard: kept", size=12, weight=600, fill=O[2])
s.add(f'<rect x="424" y="488" width="420" height="30" rx="7" fill="{L[1]}" stroke="{L[0]}" stroke-width="1.4" stroke-dasharray="5 4"/>')
s.text(634, 503, "never heard: dropped", size=12, weight=600, fill=L[2])
s.add(f'<line x1="422" y1="480" x2="422" y2="526" stroke="{PALETTE["fail"][0]}" stroke-width="2.2"/>')
s.text(430, 474, "playback stopped at 2.4 s", size=11.5, weight=600, fill=PALETTE["fail"][2], anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/11-multimodal-ai/q10-barge-in.svg")
