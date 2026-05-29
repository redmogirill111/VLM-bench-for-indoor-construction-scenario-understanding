"""
VideoLLaMA3-7B 独立推理脚本
使用独立的 videollama3_env 环境 (transformers 4.46.3)

用法:
  D:\miniconda\envs\videollama3_env\python.exe run_videollama3_inference.py

特点:
- 仅运行 VideoLLaMA3-7B 一个模型
- 无超时限制
- 断点续传 (跳过已有输出)
- 错误自动记录
"""

import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
LOG_DIR = Path(r"H:\benchmark\workspace")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "videollama3_inference.log"


def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# VideoLLaMA3 Adapter
# ---------------------------------------------------------------------------
class VideoLLaMA3Adapter:
    """VideoLLaMA3-7B 适配器 — 原生视频, trust_remote_code"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.processor = None
        self.model_name = config["name"]
        self.model_path = config["path"]

    def load(self):
        from transformers import AutoModelForCausalLM, AutoProcessor
        log(f"[{self.model_name}] Loading model (trust_remote_code=True)...")
        t0 = time.time()

        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        log(f"[{self.model_name}] Processor loaded ({time.time()-t0:.1f}s)")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        vram = torch.cuda.max_memory_allocated() / 1024**3
        log(f"[{self.model_name}] Model loaded ({time.time()-t0:.1f}s, VRAM: {vram:.2f} GB)")

    def infer(self, video_path, prompt, temperature=0.7, max_new_tokens=1024):
        """推理入口 — 使用 conversation 模式, processor 自动处理视频"""
        do_sample = temperature > 0
        gen_kwargs = {"max_new_tokens": max_new_tokens}
        if do_sample:
            gen_kwargs.update(temperature=temperature, do_sample=True, top_p=0.9)
        else:
            gen_kwargs.update(do_sample=False)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": str(video_path), "fps": 1, "max_frames": 2, "size": 224}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor(
            conversation=conversation,
            padding=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)

        # pixel_values 需要转为 bfloat16 以匹配模型权重
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != torch.bfloat16:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        generated = output_ids[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]

    def unload(self):
        for attr in ["model", "processor"]:
            v = getattr(self, attr, None)
            if v is not None:
                del v
        self.model = self.processor = None
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 主推理流程
# ---------------------------------------------------------------------------
def main():
    config_path = Path(r"H:\benchmark\scripts\model_configs.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 仅取 VideoLLaMA3-7B 配置
    model_config = None
    for m in config["models"]:
        if m["model_type"] == "videollama3":
            model_config = m
            break

    if model_config is None:
        log("ERROR: VideoLLaMA3-7B not found in model_configs.yaml")
        return

    model_name = model_config["name"]
    model_id = model_config["id"]
    clips_dir = Path(config["data"]["clips_dir"])
    manifest_file = Path(config["data"]["manifest_file"])
    output_dir = Path(config["data"]["output_dir"]) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    max_new_tokens = config["inference"].get("max_new_tokens", 1024)

    # 加载 manifest
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    raw_clips = manifest.get("clips", [])
    clips = []
    for i, c in enumerate(raw_clips, 1):
        clips.append({
            "clip_id": i,
            "file": c.get("file", c.get("file_name", "")),
            "source": c.get("source", c.get("source_video", "")),
            "duration_sec": c.get("duration_sec", 0),
        })

    # 断点续传
    existing = list(output_dir.glob("*.json"))
    completed_clips = len(existing)
    if completed_clips >= len(clips):
        log(f"[{model_name}] SKIP: {completed_clips} files already exist")
        return

    log(f"{'='*60}")
    log(f"VideoLLaMA3-7B 独立推理")
    log(f"  Model: {model_name} (id={model_id})")
    log(f"  Path: {model_config['path']}")
    log(f"  Clips: {len(clips)}, Already done: {completed_clips}")
    log(f"  Output: {output_dir}")
    log(f"  Environment: videollama3_env (transformers 4.46.3)")
    log(f"{'='*60}")

    # 加载模型
    adapter = VideoLLaMA3Adapter(model_config)
    try:
        adapter.load()
    except Exception as e:
        log(f"[{model_name}] FAILED to load: {e}")
        traceback.print_exc()
        err_file = output_dir / "_load_error.txt"
        err_file.write_text(f"{e}\n{traceback.format_exc()}", encoding="utf-8")
        return

    load_time = time.time()

    # 加载任务定义
    sys.path.insert(0, str(Path(__file__).parent))
    from prompts import get_all_tasks

    completed = 0
    failed = 0
    skipped = completed_clips

    for clip_info in clips:
        clip_id = clip_info["clip_id"]
        clip_file = clip_info["file"]
        clip_path = clips_dir / clip_file
        out_file = output_dir / f"{model_id:05d}_{clip_id:05d}.json"

        # 断点续传
        if out_file.exists():
            skipped += 1
            continue

        if not clip_path.exists():
            log(f"[{model_name}] clip={clip_id} file not found: {clip_path}")
            failed += 1
            continue

        # 准备任务
        tasks = get_all_tasks()
        result = {
            "model_id": model_id,
            "model_name": model_name,
            "clip_id": clip_id,
            "clip_file": clip_file,
            "clip_duration_sec": clip_info.get("duration_sec", 0),
            "source_video": clip_info.get("source", ""),
            "inference_timestamp": datetime.now().isoformat(),
            "tasks": {},
            "total_latency_ms": 0,
            "peak_vram_mb": 0,
        }

        total_latency = 0
        torch.cuda.reset_peak_memory_stats()

        for task_id, task_def in tasks.items():
            runs = []
            for run_i in range(1, task_def["repeat"] + 1):
                t_start = time.time()
                try:
                    output = adapter.infer(
                        str(clip_path),
                        task_def["prompt"],
                        temperature=task_def["temperature"],
                        max_new_tokens=max_new_tokens,
                    )
                except Exception as e:
                    error_msg = f"[ERROR] {type(e).__name__}: {e}"
                    output = error_msg
                    log(f"[{model_name}] clip={clip_id} task={task_id} run={run_i} ERROR: {error_msg}")
                    if run_i == 1 and task_id == "T1":
                        traceback.print_exc()

                latency = (time.time() - t_start) * 1000
                runs.append({
                    "run_id": run_i,
                    "output": output,
                    "latency_ms": round(latency, 1),
                })
                total_latency += latency

                log(f"[{model_name}] clip={clip_id} task={task_id} run={run_i} "
                    f"done ({latency/1000:.1f}s)")

            task_result = {
                "prompt": task_def["prompt"],
                "temperature": task_def["temperature"],
                "runs": runs,
            }
            if "negative_objects" in task_def:
                task_result["negative_objects"] = task_def["negative_objects"]
            result["tasks"][task_id] = task_result

        result["total_latency_ms"] = round(total_latency, 1)
        result["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)

        # 保存结果
        try:
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            completed += 1
        except Exception as e:
            log(f"[{model_name}] Save error {out_file}: {e}")
            failed += 1

        # 进度汇报
        done_count = completed + skipped + failed
        if done_count % 50 == 0 or done_count == len(clips):
            log(f"[{model_name}] Progress: {done_count}/{len(clips)} "
                f"(done={completed}, skip={skipped}, fail={failed})")

    # 卸载模型
    adapter.unload()

    total_time = time.time() - load_time
    log(f"[{model_name}] Done: completed={completed}, skipped={skipped}, failed={failed}")
    log(f"[{model_name}] Total time: {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
