// Curvilinear SoC core: every ALU bit and every instruction decode is a dot product
// of a fixed feature map with a weight array. Weights are injected by build_sandbox.py.
const WEIGHTS = __WEIGHTS__;

function combos(n, k) {
  const r = [];
  (function rec(s, c) {
    if (c.length === k) { r.push(c.slice()); return; }
    for (let i = s; i < n; i++) { c.push(i); rec(i + 1, c); c.pop(); }
  })(0, []);
  return r;
}
const C6 = [2, 3, 4].map(k => combos(6, k));
const C4 = [2, 3, 4].map(k => combos(4, k));

function featMap(x, nShells, groups) {
  const f = [1];
  let s = 0, r2 = 0;
  for (const v of x) { s += v; r2 += v * v; }
  f.push(Math.cos(Math.PI * s), Math.sin(Math.PI * s));
  const r = Math.sqrt(r2);
  for (let k = 0; k < nShells; k++) { const d = r - Math.sqrt(k); f.push(Math.exp(-d * d / (2 * 0.35 * 0.35))); }
  for (const v of x) f.push(v);
  for (const g of groups) for (const c of g) { let p = 1; for (const i of c) p *= x[i]; f.push(p); }
  return f;
}

const OPCODES = { NOP: 0, LOAD_IMM: 1, ADD: 2, SUB: 3, AND: 4, OR: 5, XOR: 6, SLT: 7,
                  LOAD: 8, STORE: 9, BEQ: 10, BNE: 11, JUMP: 12, SHL: 13, SHR: 14, HALT: 15 };
const OPNAMES = Object.keys(OPCODES);
const ALU_OPS = { ADD: 0, SUB: 1, AND: 2, OR: 3, XOR: 4, NAND: 5, NOR: 6, SLT: 7 };
const CTRL = ["RegWrite", "ALUSrc", "MemRead", "MemWrite", "Branch", "Jump", "ALUOp0", "ALUOp1"];

class NeuralCPU {
  constructor(decimals = 2) { this.setDecimals(decimals); this.reset(); }
  setDecimals(d) {
    this.decimals = d;
    const q = w => d === null ? w : Math.round(w * 10 ** d) / 10 ** d;
    this.wOut = WEIGHTS.out.map(q);
    this.wCout = WEIGHTS.cout.map(q);
    this.wDec = WEIGHTS.dec.map(row => row.map(q));
  }
  reset() {
    this.reg = [0, 0, 0, 0]; this.ram = new Array(256).fill(0);
    this.pc = 0; this.halted = false; this.cycles = 0;
    this.energies = []; this.lastCtrl = null; this.flags = { Z: 0, C: 0, N: 0, V: 0 };
  }
  load(rom, lineMap) { this.rom = rom; this.lineMap = lineMap || []; this.reset(); }
  decode(opcode) {
    const f = featMap([(opcode >> 3) & 1, (opcode >> 2) & 1, (opcode >> 1) & 1, opcode & 1], 5, C4);
    const out = [];
    for (let j = 0; j < 8; j++) { let s = 0; for (let i = 0; i < f.length; i++) s += f[i] * this.wDec[i][j]; out.push(s >= 0.5 ? 1 : 0); }
    return out;
  }
  alu(a, b, op) {
    let c = op === 1 ? 1 : 0, out = 0; const en = [];
    for (let i = 0; i < 8; i++) {
      const ab = (a >> i) & 1, bb = (b >> i) & 1, bi = op === 1 ? 1 - bb : bb, oc = op === 1 ? 0 : op;
      const f = featMap([ab, bi, c, (oc >> 2) & 1, (oc >> 1) & 1, oc & 1], 7, C6);
      let ro = 0, rc = 0;
      for (let j = 0; j < f.length; j++) { ro += f[j] * this.wOut[j]; rc += f[j] * this.wCout[j]; }
      en.push(ro); out |= (ro >= 0.5 ? 1 : 0) << i; c = rc >= 0.5 ? 1 : 0;
    }
    out &= 255;
    const sa = a >> 7, sb = b >> 7, sr = out >> 7;
    const V = op === 0 ? (sa === sb && sr !== sa ? 1 : 0) : op === 1 ? (sa !== sb && sr !== sa ? 1 : 0) : 0;
    return { result: out, flags: { Z: out === 0 ? 1 : 0, C: op <= 1 ? c : 0, N: sr, V }, en };
  }
  step() {
    if (this.halted || this.pc >= this.rom.length) { this.halted = true; return null; }
    const inst = this.rom[this.pc], pcNow = this.pc; this.pc++; this.cycles++;
    const opcode = (inst >> 12) & 15, rd = (inst >> 10) & 3, rs = (inst >> 8) & 3, imm = inst & 255;
    const ctrlBits = this.decode(opcode); this.lastCtrl = ctrlBits;
    const a = this.reg[rd], b = ctrlBits[1] ? imm : this.reg[rs];
    const amap = { 1: "ADD", 2: "ADD", 3: "SUB", 4: "AND", 5: "OR", 6: "XOR", 7: "SLT" };
    const r = this.alu(a, b, ALU_OPS[amap[opcode] || "ADD"]);
    this.energies = r.en; this.flags = r.flags;
    switch (opcode) {
      case 1: this.reg[rd] = imm; break;
      case 2: case 3: case 4: case 5: case 6: case 7: this.reg[rd] = r.result; break;
      case 8: this.reg[rd] = this.ram[imm]; break;
      case 9: this.ram[imm] = this.reg[rs]; break;
      case 10: if (a === this.reg[rs]) this.pc = imm; break;
      case 11: if (a !== this.reg[rs]) this.pc = imm; break;
      case 12: this.pc = imm; break;
      case 13: this.reg[rd] = (this.reg[rd] << 1) & 255; break;
      case 14: this.reg[rd] = this.reg[rd] >> 1; break;
      case 15: this.halted = true; break;
    }
    return { pc: pcNow, op: OPNAMES[opcode] };
  }
}

function assemble(src) {
  const labels = {}, lines = [], lineNo = [];
  let addr = 0;
  src.split("\n").forEach((raw, n) => {
    const l = raw.split(";")[0].trim();
    if (!l) return;
    if (l.endsWith(":")) { labels[l.slice(0, -1).trim()] = addr; return; }
    lines.push(l); lineNo.push(n); addr++;
  });
  const reg = t => { const m = /^R([0-3])$/i.exec(t); if (!m) throw new Error("bad register '" + t + "'"); return +m[1]; };
  const val = t => { if (t in labels) return labels[t]; const v = Number(t); if (!Number.isInteger(v) || v < 0 || v > 255) throw new Error("bad value '" + t + "' (0..255)"); return v; };
  const rom = lines.map((l, i) => {
    try {
      const p = l.replace(/,/g, " ").split(/\s+/), name = p[0].toUpperCase();
      if (!(name in OPCODES)) throw new Error("unknown op '" + p[0] + "'");
      let d = 0, s = 0, imm = 0;
      if (name === "LOAD_IMM" || name === "LOAD") { d = reg(p[1]); imm = val(p[2]); }
      else if (["ADD", "SUB", "AND", "OR", "XOR", "SLT"].includes(name)) { d = reg(p[1]); s = reg(p[2]); }
      else if (name === "STORE") { s = reg(p[1]); imm = val(p[2]); }
      else if (name === "BEQ" || name === "BNE") { d = reg(p[1]); s = reg(p[2]); imm = val(p[3]); }
      else if (name === "JUMP") imm = val(p[1]);
      else if (name === "SHL" || name === "SHR") d = reg(p[1]);
      return (OPCODES[name] << 12) | (d << 10) | (s << 8) | imm;
    } catch (e) { throw new Error("line " + (lineNo[i] + 1) + ": " + e.message); }
  });
  return { rom, lineMap: lineNo };
}

if (typeof module !== "undefined") module.exports = { NeuralCPU, assemble, WEIGHTS };
