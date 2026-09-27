"""LLM Fundamentals Q13: self-attention: Q, K, V from the same X, a T x T score matrix, mask, softmax, weighted sum."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(880, 470, "Self-attention: each token rebuilds itself from the others", "Q, K and V all come from the same sequence; this is the only place information moves between positions.")
s.box(36, 200, 116, 76, "Input X", ["T tokens × d"], "data", mono_detail=True)
for y, lab, role in [(152, "Q = X·W_Q", "model"), (214, "K = X·W_K", "compute"), (350, "V = X·W_V", "data")]:
    s.arrow([(152, 238), (172, 238), (172, y), (184, y)], role)
    s.pill(240, y, 110, 34, lab, role, size=12, mono=True)
s.arrow([(295, 152), (338, 152)], "model")
s.arrow([(295, 214), (338, 214)], "compute")
s.box(340, 138, 180, 90, "Score all pairs", ["QKᵀ / √dₖ", "a T × T matrix"], "compute", size=13.5)
s.arrow([(520, 183), (556, 183)], "compute")

# mask + softmax grid
s.grid(560, 118, 6, 22, lambda i, j: j <= i, "model")
s.text(625, 262, "mask, softmax each row", size=11.5, weight=700, fill="#3B2596")
s.text(700, 140, "row i = token i's", size=11, fill="#667085", anchor="start")
s.text(700, 156, "weights over j", size=11, fill="#667085", anchor="start")
s.text(700, 196, "future tokens", size=11, fill="#667085", anchor="start")
s.text(700, 212, "masked out", size=11, fill="#667085", anchor="start")
s.arrow([(625, 276), (625, 316)], "model")
s.box(520, 318, 210, 66, "Weighted sum of V", ["out_i = Σ_j a_ij · v_j"], "data", size=13.5, mono_detail=True)
s.arrow([(295, 350), (518, 350)], "data")
s.arrow([(625, 384), (625, 408)], "output")
s.pill(625, 428, 272, 36, "concat heads, project with W_O", "output", size=12.5)

s.box(36, 398, 330, 56, "Cost O(T²·d)", ["double the context, 4× the attention work"], "fail", size=13)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/01-llm-fundamentals/q13-self-attention.svg")
