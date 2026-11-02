"""Builds sandbox/cpu.js and sandbox/sandbox.html from the solved Curvilinear SoC weights."""
import json
import pathlib

from concentric_torch.curvilinear_computer import CurvilinearALUSlice
from concentric_torch.curvilinear_cpu_soc import CurvilinearDecoder

HERE = pathlib.Path(__file__).parent

w_out, w_cout = CurvilinearALUSlice._solve_weights()
w_dec = CurvilinearDecoder._solve_weights()
weights = {"out": w_out.tolist(), "cout": w_cout.tolist(), "dec": w_dec.tolist()}

cpu_js = (HERE / "cpu_template.js").read_text().replace("__WEIGHTS__", json.dumps(weights))
(HERE / "cpu.js").write_text(cpu_js)

html = (HERE / "page_template.html").read_text().replace("/*__CPU__*/", cpu_js)
(HERE / "sandbox.html").write_text(html)
print("wrote cpu.js and sandbox.html:", len(html), "bytes;",
      len(weights["out"]) * 2 + len(weights["dec"]) * 8, "weights embedded")
