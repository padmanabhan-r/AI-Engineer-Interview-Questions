"""System design Q1: a real-time voice agent as a streaming loop, with barge-in from VAD to the orchestrator."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 500, "A real-time voice agent", "Every stage streams into the next; VAD keeps listening while the agent speaks, so the caller can interrupt.")

# listen lane: caller audio becomes text
s.region(24, 80, 852, 130, "LISTEN · audio in → text", "data")
s.pill(92, 158, 120, 42, "Caller", "human")
s.arrow([(152, 150), (182, 150)], "human")
s.arrow([(182, 168), (152, 168)], "output")
s.box(184, 120, 180, 76, "Media edge", ["SIP / WebRTC", "+ echo cancel"], "slate", size=13.5)
s.arrow([(364, 158), (400, 158)], "data")
s.box(402, 120, 196, 76, "VAD + end-of-turn", ["semantic turn model", "on the partial transcript"], "data", size=13.5)
s.arrow([(598, 158), (634, 158)], "data")
s.box(636, 120, 212, 76, "Streaming ASR", ["partial transcripts"], "data", size=13.5)

# respond lane: text becomes audio
s.region(24, 262, 852, 130, "RESPOND · text → audio out", "model", label_pos="bl")
s.box(636, 290, 212, 70, "Orchestrator", ["starts on a confident partial"], "compute", size=13.5)
s.arrow([(742, 196), (742, 288)], "data")
s.arrow([(636, 325), (600, 325)], "model")
s.box(402, 290, 196, 70, "LLM, streamed", ["replies 50–100 tokens"], "model", size=13.5)
s.arrow([(402, 325), (366, 325)], "model")
s.box(184, 290, 180, 70, "Streaming TTS", ["per clause"], "model", size=13.5)
s.arrow([(274, 290), (274, 198)], "model")
s.text(266, 229, "agent audio", size=11.5, fill="#3B2596", weight=600, anchor="end")

# barge-in: VAD interrupts the orchestrator
s.arrow([(560, 196), (560, 236), (690, 236), (690, 288)], "fail", dashed=True)
s.text(552, 226, "barge-in: stop TTS, flush audio", size=11.5, fill="#8E2A23", weight=600, anchor="end")

# tools below the orchestrator
s.arrow([(742, 360), (742, 412)], "amber")
s.box(636, 414, 212, 60, "Tools + human handoff", ["orders, bookings"], "amber", size=13)

# latency budget
s.text(48, 426, "Targets", size=12.5, weight=700, fill="#1F3864", anchor="start")
s.text(48, 448, "< ~800 ms  caller stops → first agent audio", size=12, fill="#344054", anchor="start")
s.text(48, 468, "< ~200 ms  agent stops talking when interrupted", size=12, fill="#344054", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q01-voice-agent.svg")
