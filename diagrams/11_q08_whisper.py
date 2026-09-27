"""Multimodal Q8: the Whisper encoder-decoder, with the shapes from the answer."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 450, "Whisper: speech to text", "An encoder-decoder transformer over 30-second log-Mel windows, trained on ~680,000 hours of weakly supervised web audio.")
Y = 150
s.pill(95, Y, 130, 40, "Audio 16 kHz", "amber", size=12)
s.arrow([(160, Y), (186, Y)], "amber")
s.box(188, 105, 200, 90, "Log-Mel window", ["30 s at 10 ms hops", "80 Mel × 3,000 frames", "(128 Mel in large-v3)"], "data", size=14, detail=11)
s.arrow([(388, Y), (416, Y)], "compute")
s.box(418, 115, 160, 70, "Conv stem", ["2 conv layers", "→ 1,500 frames"], "compute", size=14)
s.arrow([(578, Y), (606, Y)], "model")
s.box(608, 115, 150, 70, "Encoder", ["transformer"], "model", size=14)

s.arrow([(683, 185), (683, 268)], "model", dashed=True)
s.text(693, 226, "cross-attention", size=11.5, weight=600, fill=PALETTE["model"][2], anchor="start")
s.box(608, 270, 150, 64, "Decoder", ["attends to the audio"], "model", size=14)
s.box(330, 270, 220, 64, "Special tokens", ["language · task · timestamps", "task = transcribe or translate"], "slate", size=13.5, detail=11)
s.arrow([(550, 302), (606, 302)], "slate", label="steer", label_dy=-10)
s.arrow([(683, 334), (683, 362)], "output")
s.pill(683, 382, 150, 36, "Transcript", "output")

s.box(36, 262, 250, 80, "Long audio", ["sequential 30 s windows, each", "prompted with the previous text"], "slate", size=13.5, detail=11)
s.text(36, 426, "Pitfall: it hallucinates fluent text on silence and noise, so gate it with VAD and drop low-confidence segments.",
       size=11.5, fill=PALETTE["fail"][2], anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/11-multimodal-ai/q08-whisper.svg")
