"""Fine-tuning Q4: one QLoRA layer, a frozen 4-bit base dequantized on the fly plus a trainable bf16 adapter."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(880, 470, "QLoRA: a 4-bit frozen base, a bf16 adapter", "Frozen weights shrink to a quarter, putting a 7B fine-tune under about 10 GB; only the adapter learns.")
Y = 250  # the activation rail
s.pill(82, Y, 110, 40, "x · bf16", "slate", mono=True, size=12.5)
s.arrow([(137, Y), (560, Y)], "slate", head=False)
s.text(300, Y - 12, "activations", size=11, fill="#667085")

# frozen base path
s.region(150, 84, 520, 118, "FROZEN BASE", "compute", label_pos="tl")
s.cylinder(236, 150, 130, 70, "W · NF4", "compute")
s.arrow([(301, 150), (330, 150)], "compute")
s.box(332, 118, 160, 64, "Dequantize", ["one block → bf16"], "compute", size=13.5)
s.arrow([(492, 150), (518, 150)], "compute")
s.pill(560, 150, 80, 40, "W x", "compute", mono=True)
s.arrow([(560, Y), (560, 172)], "slate")

# trainable adapter path
s.region(150, 298, 520, 100, "TRAINED", "amber", label_pos="tl")
s.box(470, 318, 180, 64, "LoRA  B A x", ["bf16 adapters"], "amber", size=13.5)
s.arrow([(560, Y), (560, 316)], "slate")
s.text(200, 360, "backward: gradients for", size=11, fill="#7A5300", anchor="start")
s.text(200, 376, "activations and A, B only", size=11, fill="#7A5300", anchor="start")

# merge
s.arrow([(600, 150), (720, 150), (720, 234)], "compute")
s.arrow([(650, 350), (720, 350), (720, 266)], "amber")
s.add_node(720, Y, r=14, role="model")
s.arrow([(734, Y), (768, Y)], "model")
s.pill(818, Y, 100, 40, "y · bf16", "output", mono=True, size=12)

# the memory tricks
chips = [("NF4", "16 levels at normal quantiles, blocks of 64"), ("Double quant", "scales quantized: ~0.5 → ~0.13 bit/param"), ("Paged optimizers", "spill to CPU RAM on spikes")]
for i, (k, v) in enumerate(chips):
    x = 28 + i * 280
    s.box(x, 414, 268, 42, k, [v], "slate", size=12, detail=10.5)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/05-fine-tuning-and-model-adaptation/q04-qlora.svg")
