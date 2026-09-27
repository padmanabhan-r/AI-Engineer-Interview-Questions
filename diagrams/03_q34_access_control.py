"""RAG Q34: per-user access control enforced inside retrieval, never left to the LLM."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 530, "Access control lives in retrieval", "Permissions become a mandatory pre-filter built from the authenticated identity; the LLM only ever sees permitted chunks.")
cols = [(90, "User", "human"), (270, "RAG service", "compute"), (450, "Identity provider", "slate"), (630, "Search index", "data"), (800, "LLM", "model")]
for x, name, role in cols:
    s.pill(x, 104, {800: 110, 450: 172}.get(x, 148), 38, name, role, size=12.5)
    s.add(f'<line x1="{x}" y1="124" x2="{x}" y2="466" stroke="{PALETTE[role][0]}" stroke-opacity="0.35" stroke-width="2" stroke-dasharray="4 5"/>')

def msg(y, x0, x1, label, role, dashed=False):
    s.arrow([(x0, y), (x1 - (8 if x1 > x0 else -8), y)], role, dashed=dashed)
    s.text((x0 + x1) / 2, y - 13, label, size=12, fill=PALETTE[role][2], weight=600)

msg(164, 90, 270, "Question + auth token", "human")
msg(208, 270, 450, "Resolve user, groups", "compute")
msg(246, 450, 270, "User + group IDs (cached)", "slate", dashed=True)
# the filter is the key step
s.add(f'<rect x="262" y="266" width="376" height="104" rx="12" fill="{PALETTE["amber"][1]}" stroke="{PALETTE["amber"][0]}" stroke-width="1.6"/>')
msg(298, 270, 630, "Query + filter: tenant AND ACL groups", "amber")
s.text(450, 318, "pre-filter in vector and keyword legs, not post-filter", size=11, fill="#7A5300", italic=True)
msg(356, 630, 270, "Only permitted chunks", "data", dashed=True)
msg(406, 270, 800, "Prompt with permitted chunks only", "model")
msg(448, 800, 90, "Answer with citations", "output", dashed=True)
s.add('<line x1="28" y1="482" x2="852" y2="482" stroke="#E4E7EC" stroke-width="1.5"/>')
s.text(40, 506, "Never ask the LLM to withhold what it was shown: anything in the prompt is visible to the user.", size=12, fill="#8E2A23", weight=600, anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/03-retrieval-augmented-generation-rag/q34-access-control.svg")
