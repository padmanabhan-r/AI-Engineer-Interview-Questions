// Typesets every formula in topics/*.md with KaTeX and lists the ones that fail. node scripts/check-math.mjs
import { readFileSync, readdirSync } from "node:fs";
import katex from "katex";

const dir = new URL("../topics/", import.meta.url);
let bad = 0, total = 0;
for (const name of readdirSync(dir).filter((f) => f.endsWith(".md")).sort()) {
  const text = readFileSync(new URL(name, dir), "utf8");
  const found = [
    ...[...text.matchAll(/```math\n([\s\S]*?)\n\s*```/g)].map((m) => [m[1].trim(), true]),
    ...[...text.matchAll(/\$`([^`]+)`\$/g)].map((m) => [m[1], false]),
  ];
  for (const [tex, display] of found) {
    total++;
    try { katex.renderToString(tex, { displayMode: display, throwOnError: true, strict: "ignore" }); }
    catch (e) { bad++; console.log(`${name}: ${e.message.split("\n")[0]}\n    ${tex.slice(0, 160)}`); }
  }
}
console.log(`${total} formulas, ${bad} failing`);
process.exit(bad ? 1 : 0);
