#!/usr/bin/env python3
"""Compute-in-transit budget analysis: which LLM data movements could a photonic link
module do useful work on, and how much work does each need per byte?

Writes ``results.md`` and ``results.json`` beside this file. Every number in the CiT decks
comes from here.

Model shapes and link presets are imported from Disaggregated_Inference_Sim
(``disagg_sim.hardware``) so this series and the simulator cannot drift.

Conventions (stated again in results.md):
  * FLOPs of a real FFT of length n: 2.5 n log2 n (FFTW's convention), so 2.5 log2 n per value.
    A DCT-II of length n is counted the same. A separable 2-D transform of a c x c chunk costs
    2 x 2.5 log2 c per value.
  * "Budget" = the module's claimed compute rate divided by the link's line rate, in
    operations per byte. The claim is "tens of GFLOPs" per link (Kundu et al.,
    arXiv:2608.21536); we bracket it with 10 and 99 GFLOP/s. That bracket is our reading of
    "tens", not a measurement, and a reviewer of the paper notes the figure is asserted.
  * QSFP56 carries 200 Gb/s (25 GB/s). 400G and 800G rows assume the same engine per link.
  * Training traffic: BF16 gradients, 2 bytes per parameter. A ring all-reduce sends
    2 (N - 1) / N times the buffer from each of N workers.
  * Compression ratios: 2x = BF16 to FP8; 8x = BF16 to 2-bit (MuLoCo's setting);
    85x = DeMo's reported "up to 85x less data per GPU than AdamW-DDP" (ICLR 2026). These are
    the papers' figures, used here only to size transfers.
"""

from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

from disagg_sim.hardware import LINKS, LLAMA3_8B, LLAMA3_70B

HERE = Path(__file__).parent
GB = 1e9

# ─────────────────────────────────────────────────────────── budget ──
LINE_RATES = {"QSFP56 200G": 25 * GB, "400G": 50 * GB, "800G": 100 * GB}
CLAIM_GFLOPS = {"low (10 GFLOP/s)": 10e9, "high (99 GFLOP/s)": 99e9}
PUBLISHED_OPS = ("FFT", "NTT", "modular arithmetic", "MAC")


def budget(gflops: float, rate: float) -> float:
    """Operations per byte the module can apply at line rate."""
    return gflops / rate


def fft_per_value(n: int) -> float:
    return 2.5 * math.log2(n)


def dct2d_per_value(c: int) -> float:
    return 2 * 2.5 * math.log2(c)


# ─────────────────────────────────────────────────── kernel ledger ──
def kernels() -> list[dict]:
    """Work per byte of the input stream for candidate in-transit kernels (BF16 input unless
    stated). ``transform`` = FFT/DCT FLOPs (free if the optics does the transform);
    ``other`` = everything else. ``in_set``: built only from the published operations."""
    rows = []

    def add(name, phase, transform_per_value, other_per_value, bytes_per_value, in_set, note):
        rows.append(dict(kernel=name, phase=phase,
                         transform_per_byte=transform_per_value / bytes_per_value,
                         other_per_byte=other_per_value / bytes_per_value,
                         in_set=in_set, note=note))

    add("Sum two gradient streams (in-network aggregation)", "training", 0, 1, 2, True,
        "1 add per value; needs both streams at one point, so a switch, not a pluggable")
    for n in (8, 64, 1024):
        add(f"1-D FFT, length {n}, keep a fixed band", "training / inference",
            fft_per_value(n), 0, 2, True, "selection of a fixed band is a gather, no arithmetic")
    for c in (16, 64):
        add(f"2-D DCT of {c}x{c} chunks, keep a fixed band (DeMo-like)", "training",
            dct2d_per_value(c), 0, 2, True, "DeMo itself uses top-k, which needs comparisons")
    add("2-D DCT of 64x64 chunks + top-k (DeMo as published)", "training",
        dct2d_per_value(64), 2, 2, False, "top-k: about 2 comparisons per value (illustrative)")
    add("Requantise BF16 to FP8 (per-block scale)", "training / inference", 0, 2, 2, False,
        "absmax, scale, round: rounding and comparison are not in the published set")
    add("Score a KV key against the query (dot product, head dim 128)", "inference", 0, 2, 2, True,
        "2 FLOPs per value; selecting the top blocks afterwards needs comparisons")
    return rows


# ─────────────────────────────────────────────── training traffic ──
MODELS = {"Llama-3-8B": LLAMA3_8B, "Llama-3-70B": LLAMA3_70B}
TRAIN_LINKS = ["nvlink4", "ib-ndr", "eth-100g", "eth-25g"]
RATIOS = {"none": 1, "FP8 (2x)": 2, "2-bit (8x)": 8, "DeMo reported (85x)": 85}


def ring_bytes_per_worker(buffer: float, n: int) -> float:
    return 2 * (n - 1) / n * buffer


def training_traffic() -> dict:
    out = {"models": {}, "sync": []}
    for name, m in MODELS.items():
        g = 2.0 * m.params
        out["models"][name] = {"params": m.params, "grad_bytes": g,
                               "ring8": ring_bytes_per_worker(g, 8), "ring64": ring_bytes_per_worker(g, 64)}
        for lk in TRAIN_LINKS:
            link = LINKS[lk]
            for rname, r in RATIOS.items():
                out["sync"].append(dict(model=name, link=lk, link_name=link.name, gbps=link.bandwidth / GB,
                                        ratio=rname, seconds=link.transfer_time(g / r)))
    return out


# ─────────────────────────────────────────────── inference traffic ──
def inference_traffic() -> dict:
    rows = []
    for name, m in MODELS.items():
        for prompt in (2048, 32768):
            kv = m.kv_bytes_per_token * prompt
            for lk in ("nvlink4", "ib-ndr", "eth-100g", "eth-25g"):
                rows.append(dict(model=name, prompt=prompt, kv_bytes=kv, link=lk,
                                 ms=LINKS[lk].transfer_time(kv) * 1e3))
    weights = {name: dict(bytes=2.0 * m.params, pcie5_s=LINKS["pcie5"].transfer_time(2.0 * m.params))
               for name, m in MODELS.items()}
    kvpt = {name: m.kv_bytes_per_token for name, m in MODELS.items()}
    return dict(kv=rows, weights=weights, kv_per_token=kvpt)


# ─────────────────────────── fixed band against top-k (illustrative) ──
def dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    c = np.cos(np.pi * (i + 0.5) * k / n) * math.sqrt(2.0 / n)
    c[0] /= math.sqrt(2.0)
    return c


def band_vs_topk(rng: np.random.Generator, c: int = 64, keep: float = 1 / 16, chunks: int = 256) -> list[dict]:
    """Share of each chunk's energy kept by top-k against a fixed low-frequency band of the
    same size, after a 2-D DCT of c x c chunks. Synthetic signals only: the real question,
    for real momentum tensors, is what the experiment answers."""
    C = dct_matrix(c)
    k = int(round(keep * c * c))
    side = int(round(math.sqrt(k)))
    rows = []

    def smooth(shape):
        x = rng.standard_normal(shape)
        return np.cumsum(np.cumsum(x, axis=-1), axis=-2)  # 2-D random walk: energy at low frequency

    signals = {
        "white noise": lambda: rng.standard_normal((chunks, c, c)),
        "2-D random walk (smooth)": lambda: smooth((chunks, c, c)),
        "sparse spikes (1% non-zero)": lambda: rng.standard_normal((chunks, c, c)) * (rng.random((chunks, c, c)) < 0.01),
    }
    for name, make in signals.items():
        x = make()
        X = C @ x @ C.T
        e = (X ** 2).reshape(chunks, -1)
        tot = e.sum(axis=1)
        topk = np.sort(e, axis=1)[:, ::-1][:, :k].sum(axis=1) / tot
        band = (X[:, :side, :side] ** 2).reshape(chunks, -1).sum(axis=1) / tot
        rows.append(dict(signal=name, keep=k / (c * c), topk=float(topk.mean()), band=float(band.mean())))
    return rows


# ───────────────────────────────────────────────────────── report ──
def fmt(x: float, nd: int = 2) -> str:
    return f"{x:,.{nd}f}"


def main() -> None:
    rng = np.random.default_rng(20261008)
    res: dict = {"budget": [], "kernels": kernels(), "training": training_traffic(),
                 "inference": inference_traffic(), "band_vs_topk": band_vs_topk(rng)}
    for lname, rate in LINE_RATES.items():
        for cname, g in CLAIM_GFLOPS.items():
            res["budget"].append(dict(link=lname, rate=rate, claim=cname, ops_per_byte=budget(g, rate)))
    lo = budget(10e9, LINE_RATES["QSFP56 200G"])
    hi = budget(99e9, LINE_RATES["QSFP56 200G"])
    res["qsfp56_range"] = [lo, hi]
    largest_fft = {}
    for label, b in (("low", lo), ("high", hi)):
        n = 2
        while fft_per_value(n * 2) / 2 <= b:
            n *= 2
        largest_fft[label] = n if fft_per_value(n) / 2 <= b else None
    res["largest_fft_bf16"] = largest_fft

    L = []
    w = L.append
    w("# Compute-in-transit budget analysis\n")
    w(f"Generated by `analysis/cit_budget.py` ({platform.machine()}, Python {sys.version.split()[0]}, "
      f"NumPy {np.__version__}). Every number in the CiT decks comes from this file.\n")
    w("Conventions: real FFT or DCT of length n = 2.5 log2 n FLOPs per value (FFTW's convention); a 2-D c x c "
      "transform = 2 x 2.5 log2 c per value. BF16 = 2 bytes per value. Model shapes and link presets come from "
      "Disaggregated_Inference_Sim's `disagg_sim.hardware`. The module's compute claim, \"tens of GFLOPs\" per link "
      "(Kundu et al., arXiv:2608.21536), is bracketed here by 10 and 99 GFLOP/s: our reading, not a measurement.\n")

    w("## 1. The budget: operations per byte at line rate\n")
    w("| Link | Line rate (GB/s) | Claim | Operations per byte |")
    w("|---|---|---|---|")
    for r in res["budget"]:
        w(f"| {r['link']} | {fmt(r['rate'] / GB, 0)} | {r['claim']} | {fmt(r['ops_per_byte'])} |")
    w(f"\n* On a QSFP56 link the module can apply **{fmt(lo)} to {fmt(hi)} operations per byte**. "
      "If an FFT is charged against that budget, no transform fits at the low end (even length 2 costs 1.25 per "
      f"BF16 byte) and the longest that fits at the high end is length {largest_fft['high']}. "
      "If the optics performs the transform without charge, only the remaining arithmetic counts.\n")

    w("## 2. Work per byte of candidate in-transit kernels\n")
    w("*Transform* = FFT/DCT FLOPs per input byte (zero if the optics does the transform for free). "
      "*Other* = the remaining arithmetic. *In set* = built only from the published operations "
      f"({', '.join(PUBLISHED_OPS)}).\n")
    w("| Kernel | Phase | Transform / byte | Other / byte | In set | Fits low budget (transform charged) | Fits high budget (transform charged) | Fits if transform free (low) | Note |")
    w("|---|---|---|---|---|---|---|---|---|")
    for k in res["kernels"]:
        tot = k["transform_per_byte"] + k["other_per_byte"]
        k["fits_low"] = tot <= lo
        k["fits_high"] = tot <= hi
        k["fits_free_low"] = k["other_per_byte"] <= lo
        yn = lambda b: "yes" if b else "no"
        w(f"| {k['kernel']} | {k['phase']} | {fmt(k['transform_per_byte'])} | {fmt(k['other_per_byte'])} | "
          f"{yn(k['in_set'])} | {yn(k['fits_low'])} | {yn(k['fits_high'])} | {yn(k['fits_free_low'])} | {k['note']} |")
    w("\n* The verdict turns on one unknown: whether the optical transform is charged against the budget. If it is, "
      "only short transforms and light arithmetic fit; if not, every transform-based kernel fits and the remaining "
      "arithmetic decides.\n")

    t = res["training"]
    w("## 3. Training: how much a gradient or pseudo-gradient sync moves\n")
    w("| Model | Parameters | BF16 gradient (GB) | Ring all-reduce bytes per worker, 8 workers (GB) | ... 64 workers (GB) |")
    w("|---|---|---|---|---|")
    for name, m in t["models"].items():
        w(f"| {name} | {m['params']:,} | {fmt(m['grad_bytes'] / GB)} | {fmt(m['ring8'] / GB)} | {fmt(m['ring64'] / GB)} |")
    w("\nTime to move one model-sized BF16 update once over one link, by compression ratio (milliseconds):\n")
    w("| Model | Link | GB/s | none | FP8 (2x) | 2-bit (8x) | DeMo reported (85x) |")
    w("|---|---|---|---|---|---|---|")
    for name in MODELS:
        for lk in TRAIN_LINKS:
            rows = [r for r in t["sync"] if r["model"] == name and r["link"] == lk]
            w(f"| {name} | {rows[0]['link_name']} | {fmt(rows[0]['gbps'], 1)} | "
              + " | ".join(fmt(r["seconds"] * 1e3, 1) for r in rows) + " |")
    w("\n* Over a 100 GbE cross-data-centre link a Llama-3-70B update takes "
      f"{fmt([r for r in t['sync'] if r['model'] == 'Llama-3-70B' and r['link'] == 'eth-100g' and r['ratio'] == 'none'][0]['seconds'], 1)} s "
      "uncompressed: this is the traffic DiLoCo-family methods shrink, and the place compression in transit would act.\n")

    inf = res["inference"]
    w("## 4. Inference: KV hand-off and weight streaming\n")
    w("| Model | KV bytes per token |")
    w("|---|---|")
    for name, b in inf["kv_per_token"].items():
        w(f"| {name} | {b:,.0f} |")
    w("\n| Model | Prompt | KV hand-off (MB) | NVLink 4 (ms) | IB NDR (ms) | 100 GbE (ms) | 25 GbE (ms) |")
    w("|---|---|---|---|---|---|---|")
    for name in MODELS:
        for p in (2048, 32768):
            rows = [r for r in inf["kv"] if r["model"] == name and r["prompt"] == p]
            w(f"| {name} | {p:,} | {fmt(rows[0]['kv_bytes'] / 1e6, 1)} | " + " | ".join(fmt(r["ms"], 2) for r in rows) + " |")
    w("\n| Model | BF16 weights (GB) | Load over PCIe Gen5 x16 (s) |")
    w("|---|---|---|")
    for name, x in inf["weights"].items():
        w(f"| {name} | {fmt(x['bytes'] / GB)} | {fmt(x['pcie5_s'], 2)} |")

    w("\n## 5. A fixed frequency band against top-k (synthetic signals, illustrative)\n")
    w("Share of each 64 x 64 chunk's energy kept after a 2-D DCT, keeping 1/16 of the coefficients: the largest "
      "(top-k, DeMo's choice) or a fixed low-frequency block of the same size (needs no comparisons). "
      "Synthetic signals only; whether real momentum behaves like the smooth or the noisy case is what the "
      "fixed-band experiment measures.\n")
    w("| Signal | Kept fraction | Top-k energy | Fixed band energy |")
    w("|---|---|---|---|")
    for r in res["band_vs_topk"]:
        w(f"| {r['signal']} | {fmt(r['keep'], 4)} | {fmt(r['topk'], 3)} | {fmt(r['band'], 3)} |")
    w("\n* A fixed band matches top-k only when energy sits at low frequencies (the smooth case). On noise or "
      "spikes it keeps about its own share of the energy, far below top-k.\n")

    w("## 6. Headline\n")
    w(f"* A QSFP56 compute-in-transit module offers about {fmt(lo)}-{fmt(hi)} operations per byte. Summing streams "
      "(0.5 per byte) and dot-product scoring (1 per byte) fit only towards the high end of that range; transforms fit "
      "only if the optics does them without charge.")
    w("* The largest movements in LLM work are training syncs: a Llama-3-70B BF16 update is "
      f"{fmt(t['models']['Llama-3-70B']['grad_bytes'] / GB)} GB, against "
      f"{fmt(inf['kv'][0]['kv_bytes'] / 1e6, 1)} MB for a 2,048-token Llama-3-8B KV hand-off.")
    w("* A fixed frequency band is the comparison-free substitute for top-k. It works only if the momentum's "
      "energy is concentrated at low frequencies within each chunk, which the experiment must test on real tensors.")

    (HERE / "results.md").write_text("\n".join(L) + "\n")
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()
