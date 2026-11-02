const fs = require("fs");
const { NeuralCPU, assemble } = require("./cpu.js");
const html = fs.readFileSync(__dirname + "/page_template.html", "utf8");
const s = html.indexOf("function bitmapAsm"), e = html.indexOf("const cpu=new");
const PROGRAMS = new Function(html.slice(s, e) + "; return PROGRAMS;")();

function run(name, dec, maxCycles = 5000) {
  const c = new NeuralCPU(dec);
  const a = assemble(PROGRAMS[name]);
  c.load(a.rom, a.lineMap);
  while (!c.halted && c.cycles < maxCycles) c.step();
  return c;
}
let fails = 0;
const check = (label, got, want) => { const ok = JSON.stringify(got) === JSON.stringify(want); if (!ok) fails++; console.log((ok ? "PASS " : "FAIL ") + label + " got=" + JSON.stringify(got) + " want=" + JSON.stringify(want)); };

let c = run("Multiply 6 x 7 (loop)", 2); check("multiply", c.ram[16], 42);
c = run("Factorial 4! (nested loops + RAM)", 2); check("factorial", c.ram[128], 24);
c = run("Fibonacci to RAM[32..]", 2); check("fib", c.ram.slice(32, 38), [0, 1, 1, 2, 3, 5]);
c = run("Bit tricks: AND / OR / XOR / SHL", 2); check("bits", c.ram.slice(16, 20), [170 & 204, 170 | 204, 170 ^ 204, 84]);

// heart: run ~300 cycles, display must equal BIG or SMALL bitmap at a frame boundary
const hc = new NeuralCPU(2); const ha = assemble(PROGRAMS["Beating heart (animated)"]); hc.load(ha.rom);
const seen = new Set();
for (let i = 0; i < 800; i++) { hc.step(); seen.add(hc.ram.slice(192, 256).map(v => v ? "X" : ".").join("")); }
const rows = h => h.join("");
console.log("heart distinct display states seen:", seen.size, "| contains BIG:", seen.has(rows(["", ""]) ) );
const BIG = [".XX..XX.","XXXXXXXX","XXXXXXXX","XXXXXXXX",".XXXXXX.","..XXXX..","...XX...","........"].join("");
const SMALL = ["........",".XX..XX.",".XXXXXX.",".XXXXXX.","..XXXX..","...XX...","........","........"].join("");
check("heart shows BIG frame", seen.has(BIG), true); check("heart shows SMALL frame", seen.has(SMALL), true);

// precision: 0 decimals should break the ALU
c = run("Multiply 6 x 7 (loop)", 0, 300); check("0 decimals still multiplies? (expect broken)", c.ram[16] === 42, false);
c = run("Multiply 6 x 7 (loop)", 1); check("1 decimal multiply", c.ram[16], 42);
c = run("Multiply 6 x 7 (loop)", null); check("full precision multiply", c.ram[16], 42);
console.log(fails ? fails + " FAILED" : "ALL PASSED");
process.exit(fails ? 1 : 0);
