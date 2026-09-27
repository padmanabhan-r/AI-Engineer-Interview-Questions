"""Vector DBs Q2: how an embedding model turns text into one unit-length vector, with the shapes."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 430, "Text to vector: tokenize, contextualize, pool, normalize", "One context-aware vector per token, pooled into a single vector and scaled to unit length.")

def cells(x, y, n, cell, role, label=""):
    st, fi, _ = PALETTE[role]
    for j in range(n):
        shade = 0.25 + 0.6 * ((j * 7) % 5) / 4
        s.add(f'<rect x="{x + j * cell}" y="{y}" width="{cell - 2}" height="{cell - 2}" rx="3" fill="{st}" fill-opacity="{shade:.2f}" stroke="{st}" stroke-width="1"/>')

Y = 150
s.pill(84, Y, 128, 40, '"river bank"', "slate", mono=True, size=12.5)
s.arrow([(148, Y), (174, Y)], "slate")
s.box(176, Y - 36, 158, 72, "Tokenizer", ["subword IDs, truncated", "at the max length"], "slate", size=13.5)
s.arrow([(334, Y), (360, Y)], "model")
s.box(362, Y - 36, 172, 72, "Transformer layers", ["self-attention gives", "each token its context"], "model", size=13.5)
s.arrow([(534, Y), (576, Y)], "model")

# n x d per-token matrix
gx, cell = 640, 18
s.text(gx + 5 * cell, Y - 44, "per-token vectors", size=12, weight=700, fill="#3B2596")
for i, tok in enumerate(["river", "bank"]):
    cells(gx, Y - 22 + i * 24, 10, cell, "model")
    s.text(gx - 8, Y - 13 + i * 24, tok, size=11.5, fill="#3B2596", anchor="end", mono=True)
s.text(gx + 5 * cell, Y + 44, "n × d", size=12, fill="#667085", mono=True)
s.text(gx + 5 * cell, Y + 62, '"bank" here ≠ "bank" in "bank loan"', size=11, fill="#667085")

# row 2, right to left
s.arrow([(gx + 5 * cell, Y + 74), (gx + 5 * cell, 284)], "amber")
s.box(636, 286, 190, 72, "Pooling", ["mean or [CLS] for encoders,", "last token for decoders"], "amber", size=13.5)
s.arrow([(636, 322), (576, 322)], "compute")
s.box(404, 286, 170, 72, "L2 normalize", ["cosine similarity", "= dot product"], "compute", size=13.5)
s.arrow([(404, 322), (324, 322)], "output")
cells(118, 308, 10, 20, "output")
s.text(216, 294, "one vector, d dims, length 1", size=12, weight=700, fill="#275C1C")
s.text(216, 346, "1 × d", size=12, fill="#667085", mono=True)

s.text(440, 396, "Pitfalls: text past the input limit is silently dropped, and many models need query: / passage: prefixes.", size=11.5, fill="#8E2A23", weight=600)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/06-vector-databases-and-embeddings/q02-text-to-vector.svg")
