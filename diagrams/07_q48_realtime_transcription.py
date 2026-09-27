"""System design Q48: real-time transcription for many concurrent streams: a sticky gateway, cross-stream batching on GPU, partials then finals."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 530, "Real-time transcription at 20k streams", "Chunks from many streams share each GPU batch; partials stream back fast and are replaced by a final at each endpoint.")

# clients and the sticky gateway
s.box(30, 150, 110, 140, "Clients", ["calls,", "meetings,", "captions"], "human", size=13.5)
s.arrow([(140, 184), (178, 184)], "human")
s.text(159, 172, "audio", size=10.5, fill="#1F3864", weight=600)
s.arrow([(178, 262), (140, 262)], "output")
s.text(159, 250, "text", size=10.5, fill="#275C1C", weight=600)
s.box(180, 110, 146, 280, "Gateway", ["WebSocket / gRPC", "auth", "session affinity", "per-stream buffers"], "slate", size=13.5)

# GPU workers
s.region(346, 84, 530, 256, "GPU WORKERS · streams per GPU at the latency SLO", "compute")
s.arrow([(326, 170), (350, 170)], "compute")
s.hexagon(404, 170, 104, 52, "VAD", "amber", size=13)
s.text(404, 208, "drop silence", size=11, fill="#7A5300", weight=600)
s.arrow([(456, 170), (476, 170)], "compute")
st, fi, ink = PALETTE["compute"]
s.add(f'<rect x="478" y="112" width="190" height="124" rx="12" fill="{fi}" stroke="{st}" stroke-width="1.8"/>')
s.add(f'<rect x="478" y="112" width="6" height="124" rx="3" fill="{st}"/>')
s.text(576, 130, "Cross-stream batcher", size=13, weight=700, fill=ink)
s.grid(512, 144, 4, 16, lambda i, j: j == 2, role="compute")
s.grid(576, 144, 4, 16, lambda i, j: False, role="compute")
s.text(576, 222, "a batch = one chunk per stream", size=10.5, fill=ink)
s.arrow([(668, 170), (690, 170)], "model")
s.box(692, 124, 170, 92, "Streaming ASR", ["on GPU; per-stream", "decoder state"], "model", size=13.5)
s.hexagon(576, 278, 222, 46, "Lag + real-time factor", "amber", size=12)
s.add(f'<line x1="576" y1="236" x2="576" y2="255" stroke="{PALETTE["amber"][0]}" stroke-width="2" stroke-dasharray="4 3"/>')
s.text(576, 320, "backpressure: shed or use a smaller model", size=11, fill="#7A5300", weight=600)

# post-processing and the way back
s.arrow([(777, 216), (777, 346)], "model")
s.box(672, 348, 200, 64, "Post-processing", ["punctuation, vocabulary,", "diarisation"], "model", size=13.5)
s.arrow([(672, 372), (328, 372)], "output")
s.text(500, 360, "partials fast, finals at endpoints", size=11.5, fill="#275C1C", weight=600)
s.arrow([(777, 412), (777, 440)], "data")
s.cylinder(777, 474, 190, 60, "Final transcripts", "data", size=12.5)

# partial vs final
s.text(40, 432, "What the client sees", size=12, weight=700, fill="#344054", anchor="start")
s.pill(92, 470, 104, 34, "partial", "slate", size=12, mono=True)
s.arrow([(146, 470), (170, 470)], "slate", width=1.6)
s.pill(224, 470, 104, 34, "partial", "slate", size=12, mono=True)
s.arrow([(278, 470), (302, 470)], "slate", width=1.6)
s.pill(356, 470, 104, 34, "final", "output", size=12, mono=True)
s.text(420, 462, "the final replaces", size=11, fill="#344054", anchor="start")
s.text(420, 478, "the partials", size=11, fill="#344054", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q48-realtime-transcription.svg")
