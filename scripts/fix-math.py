r"""Rewrite math into GitHub's Markdown-proof forms: inline $...$ -> $`...`$, and a $$...$$ on its own
line(s) -> a ```math block. Plain $...$ is run through Markdown first on GitHub, which eats backslashes
(\{, \_, \,) and leaves "Extra open brace" errors; these two forms are passed to the math renderer untouched.
Code blocks and inline code are left alone. Idempotent. python3 scripts/fix-math.py"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"(^[ \t]*```.*?^[ \t]*```[ \t]*$)", re.S | re.M)
CODE_SPAN = re.compile(r"(`+)[^`\n]*?\1")  # math in $`...`$ form is already held aside
DISPLAY_BLOCK = re.compile(r"^([ \t]*)\$\$(.+?)\$\$[ \t]*$", re.S | re.M)
DISPLAY_INLINE = re.compile(r"\$\$(.+?)\$\$", re.S)
INLINE = re.compile(r"(?<![\\$`])\$(?![\s`$])([^$\n]+?)(?<![\s\\])\$(?![\d`$])")

def safe(tex: str) -> str:
    """GitHub's HTML sanitiser swallows a bare < or > inside math ("x_{<t}" becomes an unbalanced brace)."""
    return re.sub(r"\s*<\s*", r" \\lt ", re.sub(r"\s*>\s*", r" \\gt ", tex)).replace("  ", " ")

def fix_text(t: str) -> str:
    # Math already in GitHub form ($`...`$) is protected first, so its backticks are never mistaken for code.
    held = []
    def hold(x: str) -> str:
        held.append(x); return f"\x01{len(held) - 1}\x01"
    t = re.sub(r"\$`([^`$\n]+)`\$", lambda m: hold(f"$`{safe(m.group(1))}`$"), t)
    # Display math alone on its line(s) becomes a math block, indented like the line it replaces.
    def block(m):
        ind, body = m.group(1), " ".join(l.strip() for l in m.group(2).strip().splitlines())
        return f"{ind}```math\n{ind}{safe(body)}\n{ind}```"
    t = DISPLAY_BLOCK.sub(block, t)
    # Then inline code, then plain $...$ and inline $$...$$.
    t = CODE_SPAN.sub(lambda m: hold(m.group(0)), t)
    t = DISPLAY_INLINE.sub(lambda m: f"$`{safe(m.group(1).strip())}`$", t)
    t = INLINE.sub(lambda m: f"$`{safe(m.group(1))}`$", t)
    for _ in range(2):  # a held span can contain another placeholder
        t = re.sub(r"\x01(\d+)\x01", lambda m: held[int(m.group(1))], t)
    return t

def fence(p: str) -> str:  # a ```math block gets the sanitiser fix; code blocks are left alone
    if not p.lstrip().startswith("```math"):
        return p
    return re.sub(r"(```math\n)(.*?)(\n[ \t]*```)", lambda m: m.group(1) + "\n".join(safe(l) if l.strip() else l for l in m.group(2).split("\n")) + m.group(3), p, flags=re.S)

total = 0
for f in sorted((ROOT / "topics").glob("*.md")):
    s = f.read_text()
    parts = FENCE.split(s)
    out = "".join(fence(p) if i % 2 else fix_text(p) for i, p in enumerate(parts))
    if out != s:
        f.write_text(out)
        n = out.count("$`") + out.count("```math") - s.count("$`") - s.count("```math")
        total += n
        print(f"{f.name:50} {n} formulas rewritten")
print("total", total)
