# -*- coding: utf-8 -*-
"""
T5v2 全量重测 — 9 模型 × 1,154 片段的新 T5（存在性 + 属性双题幻觉测试）

设计（回应审稿意见 4.1 的重新设计，详见 t5v2_library.py 头注）：
  - 每片段由固定种子确定性抽取 1 个 T5B 存在性幻觉词 + 1 个 T5C 属性幻觉词，
    同一片段对所有模型呈现相同题目（跨模型公平、可复现）；
  - temperature=0、单次推理（与主实验 T5 设置一致）；
  - 两题均正确拒绝（答"否"）→ M4v2 = 1，否则 0（解析失败记 0 并保留原始回答）。

运行方式（三个 conda 环境，按模型分组）：
  D:\\miniconda\\envs\\vlm_bench\\python.exe        run_t5v2_inference.py --models Qwen2.5-VL-7B-Instruct,...
  D:\\miniconda\\envs\\eagle25\\python.exe          run_t5v2_inference.py --models Eagle2.5-8B
  D:\\miniconda\\envs\\videollama3_env\\python.exe  run_t5v2_inference.py --models VideoLLaMA3-7B

输出：H:\\benchmark\\output_t5v2\\<模型名>\\{model_id:05d}_{clip_id:05d}.json（断点续跑，
  已存在的 (模型, 片段) 自动跳过；每模型完成后写 _COMPLETED.json 标记）。
"""
import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ── 路径与编码 ──
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ffmpeg 路径（与主实验一致，三个环境通用）
ffmpeg_bin = r"D:\miniconda\envs\vlm_bench\Library\bin"
os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")

import torch

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

# ── 新版 transformers 兼容补丁（全局，等价于主实验 ovis 分支的补丁）──
# 修复 trust_remote_code 模型（InternVL2.5/3、MiniCPM-V）加载时
# AttributeError: ... no attribute 'all_tied_weights_keys' 等问题
from transformers import PreTrainedModel

if not hasattr(PreTrainedModel, "is_parallelizable"):
    PreTrainedModel.is_parallelizable = True
if not hasattr(PreTrainedModel, "_skip_keys_device_placement"):
    PreTrainedModel._skip_keys_device_placement = None
if not hasattr(PreTrainedModel, "_keep_in_fp32_modules"):
    PreTrainedModel._keep_in_fp32_modules = None
if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = {}

# 复用主实验框架（banner 打印无副作用，main 有 __main__ 保护）
from run_batch_inference import (
    model_loader_factory,
    load_clip_manifest,
    CONFIG,
    CLIPS_DIR,
    MAX_NEW_TOKENS,
)
from t5v2_library import build_prompt, parse_answers, score_m4

OUT_DIR = Path(r"H:\benchmark\output_t5v2")
TEMPERATURE = 0.0  # 与主实验 T5 一致（TEMPERATURE_SETTINGS["T5"] == 0.0）


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run_model(model_config, clips, limit):
    model_id = model_config["id"]
    name = model_config["name"]
    model_out = OUT_DIR / name
    model_out.mkdir(parents=True, exist_ok=True)

    completed_marker = model_out / "_COMPLETED.json"
    if completed_marker.exists():
        log(f"[SKIP] {name} 已完成（存在 _COMPLETED.json），跳过")
        return

    model_path = Path(model_config["path"])
    if not model_path.exists() or not (model_path / "config.json").exists():
        log(f"[ERROR] {name} 模型路径无效: {model_path}，跳过")
        return

    log(f"加载模型: {name} ({model_config['model_type']})")
    model_fn = model_loader_factory(model_config)
    if model_fn is None:
        log(f"[ERROR] {name} 加载失败，跳过")
        return

    total = len(clips)
    done = skipped = 0
    t_start = time.time()

    for clip_info in clips:
        clip_id = clip_info["id"]
        clip_file = clip_info.get("filename", "")
        out_path = model_out / f"{model_id:05d}_{clip_id:05d}.json"

        if out_path.exists():
            skipped += 1
            continue

        clip_path = CLIPS_DIR / clip_file
        if not clip_path.exists():
            log(f"  [{clip_id}/{total}] [ERROR] 视频不存在: {clip_path}")
            continue

        prompt, item_b, item_c = build_prompt(clip_id)

        try:
            torch.cuda.synchronize()
            t0 = time.time()
            output = model_fn(clip_path, prompt, TEMPERATURE)
            torch.cuda.synchronize()
            latency_s = round(time.time() - t0, 2)
        except Exception as e:
            log(f"  [{clip_id}/{total}] [ERROR] {name} 推理失败: {e}")
            traceback.print_exc()
            output = f"ERROR: {e}"
            latency_s = 0.0

        ans1, ans2 = parse_answers(output)
        m4 = score_m4(ans1, ans2)

        record = {
            "task": "T5v2",
            "model_name": name,
            "clip_id": clip_id,
            "clip_file": clip_file,
            "t5b_item": item_b,
            "t5c_item": item_c,
            "prompt": prompt,
            "temperature": TEMPERATURE,
            "output": output,
            "ans_t5b": ans1,
            "ans_t5c": ans2,
            "m4_v2": m4,
            "parse_ok": ans1 != "unparsed" and ans2 != "unparsed",
            "latency_s": latency_s,
            "inference_timestamp": datetime.now().isoformat(),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        done += 1
        n_proc = done + skipped
        if done <= 3 or done % 20 == 0 or done == total:
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - n_proc) / rate if rate > 0 and n_proc > 0 else float("nan")
            log(f"  [{clip_id}/{total}] {name} m4={m4} ({ans1}/{ans2}) "
                f"{latency_s}s | 进度 {n_proc}/{total} | 速率 {rate*3600:.0f} clip/h | "
                f"ETA {eta/3600:.1f} h")

    # 完成标记
    info = {
        "model_name": name,
        "model_id": model_id,
        "completed": done,
        "skipped": skipped,
        "total_clips": total,
        "finished_at": datetime.now().isoformat(),
        "status": "COMPLETED",
    }
    with open(completed_marker, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    log(f"[OK] {name} 完成: 新增 {done}，跳过 {skipped}，标记写入 {completed_marker}")

    del model_fn
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description="T5v2 全量重测")
    ap.add_argument("--models", required=True,
                    help="逗号分隔的模型名（须与 model_configs.yaml 的 name 一致）")
    ap.add_argument("--limit", type=int, default=0,
                    help="仅处理前 N 个片段（0=全部），用于试跑")
    args = ap.parse_args()

    wanted = [m.strip() for m in args.models.split(",") if m.strip()]

    clips = load_clip_manifest()
    if args.limit > 0:
        clips = clips[: args.limit]

    log("=" * 72)
    log(f"T5v2 全量重测 | 模型: {wanted} | 片段: {len(clips)} | 温度: {TEMPERATURE}")
    log(f"输出目录: {OUT_DIR}")
    log("=" * 72)

    all_names = [m["name"] for m in CONFIG["models"]]
    for w in wanted:
        if w not in all_names:
            log(f"[ERROR] 未知模型名: {w}（可用: {all_names}）")
            sys.exit(2)

    for model_config in CONFIG["models"]:
        if model_config["name"] in wanted:
            run_model(model_config, clips, args.limit)

    log("全部指定模型处理完毕。")


if __name__ == "__main__":
    main()
