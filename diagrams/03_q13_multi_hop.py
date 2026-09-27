"""RAG Q13: a multi-hop question answered with two dependent retrievals; hop one's answer is hop two's query."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(860, 500, "Multi-hop: the second query needs the first answer", "The bridge entity (the team) is unknown until hop one returns, so single-shot top-k cannot find hop two's chunk.")
U, P, R = 120, 430, 740
for x, name, role in [(U, "User", "human"), (P, "Planner LLM", "model"), (R, "Retriever", "data")]:
    s.pill(x, 104, 150, 38, name, role)
    s.add(f'<line x1="{x}" y1="124" x2="{x}" y2="470" stroke="{PALETTE[role][0]}" stroke-opacity="0.35" stroke-width="2" stroke-dasharray="4 5"/>')

s.region(392, 196, 440, 104, "Hop 1", "data", label_pos="tr")
s.region(392, 316, 440, 104, "Hop 2", "data", label_pos="tr")

def msg(y, x0, x1, label, role, dashed=False, mono=False):
    s.arrow([(x0, y), (x1 - (8 if x1 > x0 else -8), y)], role, dashed=dashed)
    s.text((x0 + x1) / 2, y - 13, label, size=12, fill=PALETTE[role][2], weight=600, mono=mono)

msg(164, U, P, "Who manages the team that owns payments?", "human")
msg(240, P, R, "Which team owns the payments service?", "model")
msg(282, R, P, "Chunk A: Team Atlas", "data", dashed=True)
msg(360, P, R, "Who manages Team Atlas?", "model")
msg(402, R, P, "Chunk B names the manager", "data", dashed=True)
msg(450, P, U, "Answer, citing chunks A and B", "output", dashed=True)

# the bridge: hop one's answer is substituted into hop two's query
s.arrow([(424, 288), (350, 296), (350, 350), (424, 356)], "pink", width=2.2, dashed=True, curve=True)
s.text(340, 314, "bridge entity:", size=11.5, fill=PALETTE["pink"][2], weight=700, anchor="end")
s.text(340, 331, "Team Atlas, substituted", size=11, fill=PALETTE["pink"][2], anchor="end")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/03-retrieval-augmented-generation-rag/q13-multi-hop.svg")
