"""LLM Fundamentals Q6: one forward pass of a decoder-only Transformer."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(840, 650, "One forward pass, decoder-only Transformer", "The residual stream x carries every token through L identical blocks; each sublayer reads it and adds back.")
R = 210  # the residual rail
s.pill(R, 100, 190, 36, "Token IDs  [B, T]", "slate", mono=True, size=12.5)
s.arrow([(R, 118), (R, 140)])
s.box(R - 95, 142, 190, 40, "Embedding", role="data")
s.text(R - 14, 198, "x: [B, T, d]", size=11, fill="#667085", anchor="end", mono=True)
s.rail(R, 182, 492)

s.region(92, 208, 726, 262, "× L blocks", "model", label_pos="br")
s.arrow([(R - 14, 452), (112, 452), (112, 226), (R - 14, 226)], "model", dashed=True)  # repeat for the next block
s.text(126, 339, "next block", size=11, fill="#3B2596", weight=600, anchor="start")

# attention branch
s.arrow([(R, 262), (262, 262)], "compute")
s.pill(305, 262, 84, 30, "RMSNorm", "slate", size=12)
s.arrow([(347, 262), (368, 262)], "compute")
s.box(370, 228, 240, 70, "Causal self-attention", ["softmax(QKᵀ/√dₕ + mask)·V", "RoPE on Q, K · heads → W_O"], "model")
s.arrow([(490, 298), (490, 330), (224, 330)], "model")
s.add_node(R, 330, role="model")

# causal mask
s.grid(660, 228, 5, 22, lambda i, j: j <= i, "model")
s.text(714, 350, "causal mask", size=11.5, weight=700, fill="#3B2596")
s.text(714, 366, "token t sees only ≤ t", size=11, fill="#667085")

# ffn branch
s.arrow([(R, 385), (262, 385)], "compute")
s.pill(305, 385, 84, 30, "RMSNorm", "slate", size=12)
s.arrow([(347, 385), (368, 385)], "compute")
s.box(370, 356, 240, 58, "SwiGLU feed-forward", ["d → ≈ 8/3·d → d, per position"], "amber")
s.arrow([(490, 414), (490, 440), (224, 440)], "amber")
s.add_node(R, 440, role="amber")

s.pill(R, 508, 190, 34, "Final RMSNorm", "slate", size=12.5)
s.arrow([(R, 525), (R, 546)])
s.box(R - 110, 548, 220, 52, "Unembed · W_U", ["logits  [B, T, V]"], "output", mono_detail=True)
s.arrow([(320, 562), (400, 548)], "output")
s.arrow([(320, 586), (400, 600)], "output")
s.box(402, 520, 410, 50, "Training", ["cross-entropy: position t vs token t+1, all at once"], "output", size=13)
s.box(402, 580, 410, 50, "Inference", ["sample from the last position only; append K, V to cache"], "compute", size=13)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/01-llm-fundamentals/q06-forward-pass.svg")
