"""Vector DBs Q13: CLIP-style training of multi-modal embeddings: two encoders, one batch similarity matrix."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 500, "Multi-modal embeddings: two encoders, one shared space", "Train on image-caption pairs so each true pair scores high and every other pairing in the batch scores low.")
TY, IY = 116, 236
s.pill(78, TY, 110, 40, "Caption", "compute")
s.arrow([(133, TY), (160, TY)], "compute")
s.box(162, TY - 28, 150, 56, "Text encoder", role="compute", size=13.5)
s.arrow([(312, TY), (332, TY)], "compute")
s.pill(398, TY, 128, 34, "project to d", "slate", size=12)
s.arrow([(462, TY), (590, TY), (590, 150)], "compute")

s.pill(78, IY, 110, 40, "Image", "pink")
s.arrow([(133, IY), (160, IY)], "pink")
s.box(162, IY - 28, 150, 56, "Image encoder", role="pink", size=13.5)
s.arrow([(312, IY), (332, IY)], "pink")
s.pill(398, IY, 128, 34, "project to d", "slate", size=12)
s.arrow([(462, IY), (500, IY)], "pink")

# N x N batch similarity matrix
gx, gy, c, n = 530, 176, 30, 4
s.grid(gx, gy, n, c, lambda i, j: i == j, "output")
for k in range(n):
    s.text(gx + k * c + 14, gy - 12, f"T{k + 1}", size=11, fill="#1B418C", weight=600, mono=True)
    s.text(gx - 10, gy + k * c + 14, f"I{k + 1}", size=11, fill="#8A1F58", weight=600, mono=True, anchor="end")
s.text(gx + 60, gy + n * c + 16, "batch similarity matrix", size=11.5, fill="#344054", weight=700)
s.text(680, 196, "diagonal = true pairs:", size=11.5, fill="#275C1C", weight=700, anchor="start")
s.text(680, 213, "push scores up", size=11.5, fill="#275C1C", anchor="start")
s.text(680, 246, "everything else:", size=11.5, fill="#344054", weight=700, anchor="start")
s.text(680, 263, "push scores down", size=11.5, fill="#344054", anchor="start")

s.arrow([(gx + 60, gy + n * c + 26), (gx + 60, 350)], "amber")
s.box(gx + 60 - 150, 352, 300, 50, "Symmetric contrastive loss", ["image → text and text → image"], "amber", size=13.5)

# after training
s.box(28, 424, 270, 56, "After training", ["either encoder alone maps into the space"], "output", size=13)
s.box(28, 346, 270, 60, "Uses", ["text-to-image search, zero-shot", "classification, multimodal RAG"], "slate", size=13)
st, fi, ink = PALETTE["fail"]
s.box(320, 424, 532, 56, "Pitfall: the modality gap", ["image and text vectors occupy different regions,", "so scores are not comparable across modalities"], "fail", size=13)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/06-vector-databases-and-embeddings/q13-clip-training.svg")
