# -*- coding: utf-8 -*-
"""用 T5v2 全量重测结果生成新版数据文件（不覆盖原始数据）：
1. evaluation/metrics_summary_t5v2.json  — M4 字段替换为 M4v2（chart2 数据源）
2. evaluation/per_clip_metrics_cache_t5v2.json — M4_suppression_ratio 列替换（chart11 数据源）
"""
import json
import sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

EVAL = Path(r"H:\benchmark\evaluation")
OUT_T5V2 = Path(r"H:\benchmark\output_t5v2")

MODELS = [
    ("InternVL2.5-8B", 1), ("InternVL3-8B", 2), ("Molmo2-8B", 3),
    ("MiniCPM-V-2.6-8B", 4), ("Qwen3-VL-8B-Instruct", 5),
    ("VideoLLaMA3-7B", 7), ("Ovis2-8B", 8), ("Eagle2.5-8B", 9),
    ("Qwen2.5-VL-7B-Instruct", 10),
]

# ── 读取 T5v2 per-clip m4_v2，按 clip_id → value ──
m4_by_model = {}
for name, mid in MODELS:
    per_clip = {}
    for c in range(1, 1155):
        f = OUT_T5V2 / name / f"{mid:05d}_{c:05d}.json"
        r = json.load(open(f, encoding="utf-8"))
        per_clip[r["clip_id"]] = r["m4_v2"]
    m4_by_model[name] = per_clip
    print(f"  {name}: M4v2 = {sum(per_clip.values())/len(per_clip):.4f} (n={len(per_clip)})")

# ── 1) 新 summary ──
summary = json.load(open(EVAL / "metrics_summary.json", encoding="utf-8"))
for name, _ in MODELS:
    pc = m4_by_model[name]
    summary[name]["M4_hallucination_suppression"] = round(sum(pc.values()) / len(pc), 4)
out_s = EVAL / "metrics_summary_t5v2.json"
json.dump(summary, open(out_s, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"[OK] {out_s.name}")

# ── 2) 新 per-clip cache（v3：M4 列替换，其余与 v2 相同）──
cache = json.load(open(EVAL / "per_clip_metrics_cache_v2.json", encoding="utf-8"))
clip_ids = cache["clip_ids"]                     # 顺序基准
cache_models = cache["models"]                   # 顺序基准
for mi, name in enumerate(cache_models):
    pc = m4_by_model[name]
    cache["M4_suppression_ratio"][mi] = [pc[cid] for cid in clip_ids]
out_c = EVAL / "per_clip_metrics_cache_t5v2.json"
json.dump(cache, open(out_c, "w", encoding="utf-8"), ensure_ascii=False)
print(f"[OK] {out_c.name}")

# 校验：cache 列均值 vs summary
for mi, name in enumerate(cache_models[:3]):
    col = cache["M4_suppression_ratio"][mi]
    print(f"  check {name}: cache mean = {sum(col)/len(col):.4f}, summary = {summary[name]['M4_hallucination_suppression']:.4f}")
print("done.")
