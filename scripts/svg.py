"""A small SVG kit for the answer illustrations. Every diagram is a Python script that builds one
self-contained SVG (its own light card, system fonts, no external assets), so it renders the same on
GitHub in light and dark mode.

Shape carries meaning, so a diagram reads without a legend:
  pill      a step or operation            box       a component (title + detail lines)
  cylinder  stored data                    hexagon   a gate or check that can block
  diamond   a decision                     add       a residual add / merge point
Colour = role: model (violet), data (teal), compute (blue), ffn/warn (amber), output (green),
human (navy), fail (red), neutral (slate).
"""
from html import escape

PALETTE = {
    #            stroke     fill       ink
    "model":   ("#6D4AE0", "#F1ECFF", "#3B2596"),
    "data":    ("#0E8C7E", "#E3F6F2", "#0A5A51"),
    "compute": ("#2F6FE4", "#E7F0FF", "#1B418C"),
    "amber":   ("#C98A06", "#FFF4D6", "#7A5300"),
    "output":  ("#3E8E2F", "#E8F5E1", "#275C1C"),
    "human":   ("#1F3864", "#E8EEF8", "#1F3864"),
    "fail":    ("#D0443A", "#FDE8E6", "#8E2A23"),
    "slate":   ("#667085", "#F2F4F7", "#344054"),
    "pink":    ("#D6408F", "#FDE9F3", "#8A1F58"),
}
SANS = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "SFMono-Regular, Menlo, Consolas, monospace"
INK, MUTED = "#151A23", "#667085"


def fits(label: str, size: float, width: float, weight: int = 400):
    """Warn when a label is likely wider than its shape (a rough per-character estimate for system sans)."""
    est = max(len(l) for l in str(label).split("\n")) * size * (0.58 if weight >= 600 else 0.53)
    if est > width - 16:
        import sys
        print(f"  warning: text may overflow ({est:.0f}px > {width - 16:.0f}px): {label!r}", file=sys.stderr)


class Svg:
    def __init__(self, w: int, h: int, title: str, subtitle: str = ""):
        self.w, self.h, self.parts, self.markers = w, h, [], set()
        self.title, self.subtitle = title, subtitle

    def add(self, s: str):
        self.parts.append(s)

    # ---- text ----
    def text(self, x, y, s, size=13, weight=400, fill=INK, anchor="middle", mono=False, italic=False, opacity=1):
        lines = str(s).split("\n")
        y0 = y - (len(lines) - 1) * size * 0.62
        for i, line in enumerate(lines):
            self.add(f'<text x="{x:.1f}" y="{y0 + i * size * 1.25:.1f}" font-size="{size}" font-weight="{weight}" '
                     f'fill="{fill}" text-anchor="{anchor}" dominant-baseline="middle" '
                     f'font-family="{MONO if mono else SANS}"{" font-style=\"italic\"" if italic else ""} opacity="{opacity}">'
                     f'{escape(line)}</text>')

    # ---- shapes ----
    def pill(self, cx, cy, w, h, label, role="slate", size=13, weight=600, mono=False):
        s, f, ink = PALETTE[role]
        fits(label, size, w - h * 0.6, weight)
        self.add(f'<rect x="{cx - w/2:.1f}" y="{cy - h/2:.1f}" width="{w}" height="{h}" rx="{h/2}" fill="{f}" stroke="{s}" stroke-width="1.6"/>')
        self.text(cx, cy, label, size=size, weight=weight, fill=ink, mono=mono)

    def box(self, x, y, w, h, title, lines=(), role="compute", size=14, detail=11.5, mono_detail=False):
        s, f, ink = PALETTE[role]
        fits(title, size, w, 700)
        for l in lines: fits(l, detail, w)
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{f}" stroke="{s}" stroke-width="1.8"/>')
        self.add(f'<rect x="{x}" y="{y}" width="6" height="{h}" rx="3" fill="{s}"/>')
        n = len(lines)
        ty = y + h / 2 - (n * detail * 1.35) / 2 if n else y + h / 2
        self.text(x + w / 2 + 3, ty, title, size=size, weight=700, fill=ink)
        for i, l in enumerate(lines):
            self.text(x + w / 2 + 3, ty + size * 0.95 + i * detail * 1.35, l, size=detail, fill=ink, opacity=0.85, mono=mono_detail)

    def cylinder(self, cx, cy, w, h, label, role="data", size=13):
        s, f, ink = PALETTE[role]
        rx, ry = w / 2, 9
        top, bot = cy - h / 2 + ry, cy + h / 2 - ry
        self.add(f'<path d="M{cx - rx},{top} L{cx - rx},{bot} A{rx},{ry} 0 0 0 {cx + rx},{bot} L{cx + rx},{top}" fill="{f}" stroke="{s}" stroke-width="1.8"/>')
        self.add(f'<ellipse cx="{cx}" cy="{top}" rx="{rx}" ry="{ry}" fill="{f}" stroke="{s}" stroke-width="1.8"/>')
        self.text(cx, cy + 5, label, size=size, weight=600, fill=ink)

    def hexagon(self, cx, cy, w, h, label, role="amber", size=13):
        s, f, ink = PALETTE[role]
        d = h / 2
        pts = [(cx - w/2 + d, cy - h/2), (cx + w/2 - d, cy - h/2), (cx + w/2, cy), (cx + w/2 - d, cy + h/2), (cx - w/2 + d, cy + h/2), (cx - w/2, cy)]
        self.add(f'<polygon points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" fill="{f}" stroke="{s}" stroke-width="1.8"/>')
        self.text(cx, cy, label, size=size, weight=600, fill=ink)

    def diamond(self, cx, cy, w, h, label, role="amber", size=12.5):
        s, f, ink = PALETTE[role]
        self.add(f'<polygon points="{cx},{cy - h/2} {cx + w/2},{cy} {cx},{cy + h/2} {cx - w/2},{cy}" fill="{f}" stroke="{s}" stroke-width="1.8"/>')
        self.text(cx, cy, label, size=size, weight=600, fill=ink)

    def add_node(self, cx, cy, r=12, role="human"):
        s = PALETTE[role][0]
        self.add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#FFFFFF" stroke="{s}" stroke-width="2"/>')
        self.add(f'<path d="M{cx - r*0.5},{cy} H{cx + r*0.5} M{cx},{cy - r*0.5} V{cy + r*0.5}" stroke="{s}" stroke-width="2.2" stroke-linecap="round"/>')

    def region(self, x, y, w, h, label="", role="model", dashed=True, label_pos="tl"):
        s, f, ink = PALETTE[role]
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{f}" fill-opacity="0.45" stroke="{s}" stroke-opacity="0.6" stroke-width="1.5"{dash}/>')
        if label:
            lx, anchor = (x + 16, "start") if label_pos in ("tl", "bl") else (x + w - 16, "end")
            ly = y + 18 if label_pos in ("tl", "tr") else y + h - 16
            self.text(lx, ly, label, size=12.5, weight=700, fill=ink, anchor=anchor)

    # ---- connectors ----
    def _marker(self, color):
        mid = "m" + color.lstrip("#")
        if mid not in self.markers:
            self.markers.add(mid)
        return mid

    def arrow(self, pts, role="slate", width=2, dashed=False, label="", label_at=0.5, label_dy=-10, head=True, curve=False):
        color = PALETTE[role][0]
        mid = self._marker(color)
        if curve and len(pts) == 4:
            (x0, y0), (c1x, c1y), (c2x, c2y), (x1, y1) = pts
            d = f"M{x0},{y0} C{c1x},{c1y} {c2x},{c2y} {x1},{y1}"
        else:
            d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        end = f' marker-end="url(#{mid})"' if head else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{dash}{end}/>')
        if label:
            (ax, ay), (bx, by) = (pts[0], pts[-1]) if not curve else (pts[0], pts[3])
            self.text(ax + (bx - ax) * label_at, ay + (by - ay) * label_at + label_dy, label, size=11.5, fill=PALETTE[role][2], weight=600)

    def rail(self, x, y0, y1, color="#9DB7E8", width=12):
        self.add(f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y1}" stroke="{color}" stroke-width="{width}" stroke-linecap="round" opacity="0.55"/>')

    def grid(self, x, y, n, cell, filled, role="model"):
        s, f, _ = PALETTE[role]
        for i in range(n):
            for j in range(n):
                on = filled(i, j)
                self.add(f'<rect x="{x + j*cell}" y="{y + i*cell}" width="{cell - 2}" height="{cell - 2}" rx="3" '
                         f'fill="{s if on else "#F2F4F7"}" fill-opacity="{0.8 if on else 1}" stroke="{s if on else "#D0D5DD"}" stroke-width="1"/>')

    # ---- output ----
    def render(self) -> str:
        defs = "".join(
            f'<marker id="{m}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="#{m[1:]}"/></marker>' for m in sorted(self.markers))
        head = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" role="img" aria-label="{escape(self.title)}">',
            f'<defs>{defs}<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#F6F7FB"/></linearGradient></defs>',
            f'<rect x="1" y="1" width="{self.w - 2}" height="{self.h - 2}" rx="18" fill="url(#bg)" stroke="#E4E7EC" stroke-width="1.5"/>',
        ]
        t = []
        if self.title:
            t.append(f'<text x="28" y="36" font-size="18" font-weight="700" fill="{INK}" font-family="{SANS}">{escape(self.title)}</text>')
        if self.subtitle:
            t.append(f'<text x="28" y="58" font-size="12.5" fill="{MUTED}" font-family="{SANS}">{escape(self.subtitle)}</text>')
        return "\n".join(head + t + self.parts + ["</svg>"]) + "\n"

    def save(self, path):
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.render())
        return p
