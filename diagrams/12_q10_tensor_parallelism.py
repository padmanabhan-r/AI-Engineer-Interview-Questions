"""Infrastructure Q10: tensor parallelism on an MLP, A split by columns, B by rows, one all-reduce."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 480, "Tensor parallelism: every GPU computes a slice of every layer", "In the MLP, split A by columns and B by rows: the only communication is one all-reduce summing the partial outputs.")

def matrix(x, y, part, role, label):
    """A 56x56 weight matrix outline with this GPU's slice filled: part is 'left', 'right', 'top' or 'bottom'."""
    c, f, ink = PALETTE[role]
    s.add(f'<rect x="{x}" y="{y}" width="56" height="56" rx="4" fill="#FFFFFF" stroke="#98A2B3" stroke-width="1.3" stroke-dasharray="3 3"/>')
    sx, sy, sw, sh = {"left": (x, y, 28, 56), "right": (x + 28, y, 28, 56), "top": (x, y, 56, 28), "bottom": (x, y + 28, 56, 28)}[part]
    s.add(f'<rect x="{sx}" y="{sy}" width="{sw}" height="{sh}" rx="4" fill="{c}" fill-opacity="0.8" stroke="{c}" stroke-width="1.3"/>')
    s.text(x + 28, y + 72, label, size=11.5, fill=ink, weight=600)

s.box(28, 222, 104, 56, "X", ["replicated"], "slate", size=14)
for i, (y0, a_part, b_part, role) in enumerate([(84, "left", "top", "compute"), (266, "right", "bottom", "model")]):
    k = i + 1
    cy = y0 + 76
    s.region(160, y0, 492, 150, f"GPU {k}", role, label_pos="tl")
    s.arrow([(80, 278 if k == 2 else 222), (80, cy), (184, cy)], "slate")
    matrix(186, cy - 28, a_part, role, f"A{k}: columns")
    s.arrow([(242, cy), (262, cy)], role)
    s.pill(330, cy, 132, 38, f"GeLU(X·A{k})", role, size=12.5, mono=True)
    s.text(330, cy + 34, "local, no comms", size=11, fill="#667085")
    s.arrow([(396, cy), (416, cy)], role)
    matrix(418, cy - 28, b_part, role, f"B{k}: rows")
    s.arrow([(474, cy), (494, cy)], role)
    s.pill(566, cy, 136, 38, f"partial Y{k}", role, size=12.5)
    s.arrow([(634, cy), (716, cy), (716, 250 + (-16 if k == 1 else 16))], role)

s.add_node(716, 250, r=16, role="amber")
s.arrow([(732, 250), (762, 250)], "amber")
s.box(764, 222, 100, 56, "Y", ["replicated"], "output", size=14)
s.text(730, 300, "all-reduce", size=12, weight=700, fill="#7A5300", anchor="start")
s.text(730, 317, "Y = Σ Yᵢ", size=12, fill="#7A5300", anchor="start", mono=True)
s.text(28, 442, "Attention: heads split across GPUs, output projection row-parallel → two all-reduces per layer.", size=11.5, fill="#344054", anchor="start")
s.text(28, 462, "Each GPU reads 1/t of the weights per token, so decode speeds up nearly t×; keep TP inside one NVLink node.", size=11.5, fill="#344054", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/12-ai-infrastructure-and-scalability/q10-tensor-parallelism.svg")
