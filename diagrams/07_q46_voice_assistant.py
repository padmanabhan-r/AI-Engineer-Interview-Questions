"""System design Q46: a device voice assistant: wake word and simple commands on the device, open requests in the cloud."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 560, "A voice assistant", "Nothing leaves the device until the wake word fires; simple commands stay local, open requests go to an LLM with tools.")

# on-device column
s.region(24, 78, 420, 460, "ON DEVICE · private by default", "data")
s.pill(180, 130, 140, 40, "Microphone", "slate")
s.arrow([(180, 150), (180, 178)], "slate")
s.hexagon(180, 208, 190, 56, "Wake word", "amber", size=13.5)
s.text(288, 190, "tiny always-on model", size=11, fill="#7A5300", weight=600, anchor="start")
s.text(288, 208, "audio before it is", size=11, fill="#7A5300", anchor="start")
s.text(288, 224, "never uploaded", size=11, fill="#7A5300", anchor="start")
s.arrow([(180, 236), (180, 266)], "compute")
s.pill(180, 288, 196, 42, "VAD + endpointing", "compute", size=12.5)
s.arrow([(180, 309), (180, 334)], "compute")
s.box(72, 336, 216, 72, "On-device ASR + intent", ["timers, volume, lights;", "works offline"], "compute", size=13.5)
s.arrow([(180, 408), (180, 450)], "output")
s.text(190, 430, "handled", size=11.5, fill="#275C1C", weight=600, anchor="start")
s.box(88, 452, 184, 56, "Device action", [], "output", size=13.5)

# cloud column
s.region(460, 78, 416, 460, "CLOUD · open requests", "compute")
s.arrow([(288, 372), (532, 372)], "model")
s.text(410, 360, "complex", size=11.5, fill="#3B2596", weight=600)
s.box(534, 336, 290, 72, "Cloud ASR + LLM with tools", ["device state injected as context"], "model", size=13.5)
s.arrow([(620, 336), (620, 280)], "fail")
s.hexagon(620, 252, 250, 56, "Confirm if risky\ndoors, purchases, messages", "fail", size=12)
s.arrow([(620, 224), (620, 176)], "amber")
s.box(534, 116, 290, 58, "Tools", ["smart home, calendar, music, search"], "amber", size=13.5)
s.arrow([(780, 174), (780, 334)], "amber")
s.text(790, 254, "results", size=11.5, fill="#7A5300", weight=600, anchor="start")
s.arrow([(680, 408), (680, 458)], "model")
s.pill(680, 480, 196, 42, "Streaming TTS", "model", size=12.5)
s.arrow([(272, 480), (580, 480)], "output")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q46-voice-assistant.svg")
