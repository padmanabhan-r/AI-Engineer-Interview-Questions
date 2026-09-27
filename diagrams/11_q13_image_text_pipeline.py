"""Multimodal Q13: a system that processes images and text, as a routed and validated pipeline."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 420, "Images + text: a pipeline, not one call", "Route each image to the cheapest capable component, then validate before anything is automated.")
Y = 210
s.box(24, Y - 30, 100, 60, "Inputs", ["photos, text"], "human", size=13.5)
s.arrow([(124, Y), (142, Y)], "human")
s.box(144, Y - 35, 160, 70, "Normalize", ["fix EXIF rotation,", "strip location, resize"], "slate", size=13.5)
s.diamond(352, Y, 94, 90, "Route", "amber", size=13)
A = PALETTE["amber"][2]
s.arrow([(352, 165), (352, 118), (444, 118)], "amber")
s.text(400, 107, "document", size=11, weight=600, fill=A)
s.arrow([(399, Y), (444, Y)], "amber")
s.text(421, Y - 11, "simple", size=11, weight=600, fill=A)
s.arrow([(352, 255), (352, 302), (444, 302)], "amber")
s.text(400, 291, "scene", size=11, weight=600, fill=A)
s.box(446, 90, 150, 56, "OCR + layout", ["text-heavy images"], "compute", size=13.5)
s.box(446, 182, 150, 56, "Classifier", ["small; simple checks"], "data", size=13.5)
s.box(446, 270, 150, 64, "VLM", ["JSON schema, enums,", "a “not visible” option"], "model", size=13.5, detail=11)
# collector bus into validation
for y in (118, Y, 302):
    s.arrow([(596, y), (616, y)], "slate", head=False)
s.arrow([(616, 118), (616, 302)], "slate", head=False)
s.arrow([(616, Y), (668, Y)], "slate")
s.hexagon(740, Y, 140, 70, "Validate vs\ntext + rules", "amber", size=12.5)
s.arrow([(740, 175), (740, 142)], "output")
s.text(750, 160, "confident", size=11, weight=600, fill=PALETTE["output"][2], anchor="start")
s.box(650, 86, 180, 54, "Automated decision", role="output", size=13.5)
s.arrow([(740, 245), (740, 278)], "human")
s.text(750, 262, "uncertain", size=11, weight=600, fill=PALETTE["human"][2], anchor="start")
s.box(650, 280, 180, 54, "Human review", role="human", size=13.5)

s.text(36, 376, "Cross-check the modalities: the claim text says rear bumper, the photo shows the front, so flag it.",
       size=11.5, fill="#344054", anchor="start")
s.text(36, 396, "Trace every call: image version, prompt, model version and output; score each field on a labeled set.",
       size=11.5, fill="#667085", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/11-multimodal-ai/q13-image-text-pipeline.svg")
