"""Vector DBs Q15: what quantization saves, and the search-then-rescore pattern that recovers the accuracy."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 440, "Quantized embeddings: search small, rescore exact", "Fewer bits per vector cut memory and distance cost; rescoring the top candidates with full vectors wins back the ranking.")

# storage bars for the answer's example
s.region(24, 80, 832, 168, "100M × 1536 vectors, stored as", "slate", dashed=False)
scale = 520 / 614
rows = [("float32", "4 bytes per dimension", 614, "614 GB", "slate"),
        ("int8", "4× smaller", 154, "154 GB", "compute"),
        ("PQ, m = 96", "1 byte per subvector", 9.6, "≈ 9.6 GB", "output")]
for i, (name, how, gb, lab, role) in enumerate(rows):
    y = 122 + i * 40
    st, fi, ink = PALETTE[role]
    s.text(48, y, name, size=12.5, weight=700, fill=ink, anchor="start", mono=True)
    s.text(48, y + 15, how, size=10.5, fill="#667085", anchor="start")
    w = max(gb * scale, 4)
    s.add(f'<rect x="200" y="{y - 11}" width="{w:.1f}" height="22" rx="4" fill="{st}" fill-opacity="0.85"/>')
    s.text(200 + w + 10, y, lab, size=12.5, weight=700, fill=ink, anchor="start")

# two-stage search
Y = 316
s.pill(76, Y, 96, 40, "Query", "human")
s.arrow([(124, Y), (150, Y)], "compute")
s.box(152, Y - 32, 176, 64, "ANN over quantized", ["vectors, in RAM"], "compute", size=13)
s.arrow([(328, Y), (369, Y)], "compute")
s.pill(446, Y, 150, 50, "top 100–500\ncandidates", "amber", size=12)
s.arrow([(521, Y), (562, Y)], "data")
s.box(564, Y - 32, 176, 64, "Rescore with full", ["vectors from disk"], "data", size=13)
s.arrow([(740, Y), (768, Y)], "output")
s.pill(812, Y, 84, 40, "Top 10", "output", size=12.5)

s.text(440, 384, "int8 is nearly free: make it the default at scale.", size=12, fill="#1B418C", weight=700)
s.text(440, 406, "Use binary and aggressive PQ only with full-precision rescoring of the top candidates.", size=11.5, fill="#8E2A23", weight=600)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/06-vector-databases-and-embeddings/q15-quantization.svg")
