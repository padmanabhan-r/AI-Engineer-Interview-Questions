"""System design Q13: a music generation service, plan the structure, generate codec tokens per section, then check rights."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 520, "A music generation service", "Plan the song's sections, generate audio tokens section by section, decode, then check rights before it ships.")

# request
s.pill(126, 140, 196, 42, "Style prompt + lyrics", "human", size=12.5)
s.arrow([(224, 140), (246, 140)], "human")
s.hexagon(346, 140, 196, 62, "Filters", "fail", size=13.5)
s.text(346, 188, "artist names, policy", size=11.5, fill="#8E2A23", weight=600)
s.arrow([(444, 140), (470, 140)], "fail")
s.box(472, 102, 240, 76, "Structure plan", ["sections + timing,", "lyrics with section markers"], "compute", size=14)

# generator: one block per section, each conditioned on the audio before it
s.region(24, 222, 852, 132, "AUDIO GENERATOR · codec tokens (RVQ)", "model", label_pos="bl")
s.arrow([(592, 178), (592, 210), (120, 210), (120, 244)], "compute")
col, fill, ink = PALETTE["model"]
for i, name in enumerate(["verse", "chorus", "verse", "chorus"]):
    x = 50 + i * 208
    s.add(f'<rect x="{x}" y="246" width="164" height="60" rx="10" fill="{fill}" stroke="{col}" stroke-width="1.6"/>')
    s.text(x + 82, 262, name, size=13, weight=700, fill=ink)
    for r in range(3):  # several codebooks per frame
        for c in range(9):
            s.add(f'<rect x="{x + 18 + c * 14.5}" y="{274 + r * 8}" width="12" height="6" rx="1.5" fill="{col}" fill-opacity="{0.85 - r * 0.25}"/>')
    if i < 3:
        s.arrow([(x + 164, 276), (x + 206, 276)], "model")
s.text(706, 326, "previous audio as context, so choruses repeat", size=11.5, fill=ink, weight=600)

# decode and ship
s.arrow([(415, 354), (415, 392)], "model")
s.box(330, 394, 170, 64, "Codec decoder", ["tokens → waveform"], "model", size=14)
s.arrow([(330, 426), (262, 426)], "output")
s.pill(170, 426, 180, 42, "Streamed preview", "output", size=12.5)
s.text(170, 462, "within ~20 s", size=11.5, fill="#275C1C", weight=600)
s.arrow([(500, 426), (528, 426)], "model")
s.hexagon(620, 426, 180, 64, "Catalogue check\n+ watermark", "fail", size=12.5)
s.arrow([(710, 426), (738, 426)], "fail")
s.cylinder(800, 426, 120, 70, "Storage + CDN", "data", size=12)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q13-music-generation.svg")
