"""
Eagle2.5-8B 单独推理脚本
使用 eagle25 conda 环境 (transformers==4.55.4) 运行
"""

import os
import sys
import json
import time
import random
import traceback
from pathlib import Path
from datetime import datetime

# 路径设置
SCRIPTS_DIR = Path(r"H:\benchmark\scripts")
sys.path.insert(0, str(SCRIPTS_DIR))

import torch
import yaml

# 优化
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

from prompts import get_all_prompts, TEMPERATURE_SETTINGS, REPEAT_COUNTS, NEGATIVE_OBJECTS

# 加载配置
CONFIG_PATH = SCRIPTS_DIR / "model_configs.yaml"
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

CLIPS_DIR = Path(CONFIG['data']['clips_dir'])
MANIFEST_FILE = Path(CONFIG['data']['manifest_file'])
OUTPUT_DIR = Path(CONFIG['data']['output_dir'])
MAX_NEW_TOKENS = CONFIG['inference']['max_new_tokens']
GPU_ID = CONFIG['inference']['gpu_id']

# ffmpeg 路径
ffmpeg_bin = r"D:\miniconda\envs\vlm_bench\Library\bin"
os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")

# Eagle2.5-8B 模型配置
EAGLE_CONFIG = {
    "id": 9,
    "name": "Eagle2.5-8B",
    "path": r"H:\benchmark\models\Eagle2.5-8B",
    "model_type": "eagle25",
    "dtype": "bfloat16",
}
MODEL_ID = EAGLE_CONFIG["id"]
MODEL_NAME = EAGLE_CONFIG["name"]
MODEL_PATH = EAGLE_CONFIG["path"]

print("=" * 80)
print("Eagle2.5-8B 专用推理脚本")
print("=" * 80)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"GPU: {torch.cuda.get_device_name(GPU_ID)}")
print(f"PyTorch: {torch.__version__}")
print(f"transformers: {__import__('transformers').__version__}")
print(f"decord: {__import__('decord').__version__}")
print("=" * 80)


def load_clip_manifest():
    if not MANIFEST_FILE.exists():
        clips = list(CLIPS_DIR.glob("*.mp4"))
        return [{"id": i+1, "filename": clip.name} for i, clip in enumerate(sorted(clips))]
    with open(MANIFEST_FILE, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    if isinstance(manifest, dict):
        clips = manifest.get('clips', [])
        for i, clip in enumerate(clips):
            clip['id'] = i + 1
            clip['filename'] = clip.get('file', '')
    else:
        clips = manifest
    print(f"[OK] 加载 manifest: {len(clips)} 个视频片段")
    return clips


def get_video_duration(video_path):
    import subprocess
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return float(result.stdout.strip())
    except:
        return 0.0


def load_model():
    from transformers import AutoProcessor, AutoModel

    device = f"cuda:{GPU_ID}"

    print(f"\n加载模型: {MODEL_NAME}")
    print(f"  路径: {MODEL_PATH}")

    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval().to(device)

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, trust_remote_code=True, use_fast=True,
    )
    processor.tokenizer.padding_side = "left"

    print("[OK] 模型加载完成")

    def infer_fn(video_path, prompt, temperature):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path), "nframes": 4},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_list = [processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )]
        image_inputs, video_inputs, video_kwargs = processor.process_vision_info(
            messages, return_video_kwargs=True
        )
        inputs = processor(
            text=text_list,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            videos_kwargs=video_kwargs,
        )
        inputs = inputs.to(device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
            )

        output = processor.batch_decode(
            generated_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output

    return infer_fn


def run_inference_for_clip(model_fn, clip_info):
    clip_file = clip_info.get("filename", "")
    clip_path = CLIPS_DIR / clip_file
    clip_id = clip_info.get("id", 0)

    if not clip_path.exists():
        print(f"  [ERROR] 视频文件不存在: {clip_path}")
        return None

    result = {
        "model_name": MODEL_NAME,
        "clip_id": clip_id,
        "clip_file": clip_file,
        "clip_duration_sec": get_video_duration(clip_path),
        "inference_timestamp": datetime.now().isoformat(),
        "tasks": {}
    }

    total_latency = 0
    peak_vram = 0

    for task_name, prompt_template in get_all_prompts().items():
        if task_name == "T5":
            selected_objects = random.sample(NEGATIVE_OBJECTS, 2)
            prompt = f"视频中是否包含以下物体？[{', '.join(selected_objects)}]。请逐项回答是/否并说明理由。"
        else:
            prompt = prompt_template

        temperature = TEMPERATURE_SETTINGS[task_name]
        repeat_count = REPEAT_COUNTS[task_name]

        result["tasks"][task_name] = {
            "prompt": prompt,
            "temperature": temperature,
            "runs": []
        }

        for run_id in range(1, repeat_count + 1):
            try:
                torch.cuda.synchronize()
                t0 = time.time()

                output = model_fn(clip_path, prompt, temperature)

                torch.cuda.synchronize()
                latency_ms = (time.time() - t0) * 1000
                vram_after = torch.cuda.max_memory_allocated() / 1024**2

                total_latency += latency_ms
                peak_vram = max(peak_vram, vram_after)

                result["tasks"][task_name]["runs"].append({
                    "run_id": run_id,
                    "output": output,
                    "latency_ms": round(latency_ms, 2),
                    "vram_mb": round(vram_after, 2)
                })
                print(f"    {task_name} run {run_id}/{repeat_count}: {latency_ms:.0f}ms")

            except Exception as e:
                print(f"    [ERROR] {task_name} run {run_id} 失败: {e}")
                traceback.print_exc()
                result["tasks"][task_name]["runs"].append({
                    "run_id": run_id,
                    "output": f"ERROR: {str(e)}",
                    "latency_ms": 0,
                    "vram_mb": 0
                })

    result["total_latency_ms"] = round(total_latency, 2)
    result["peak_vram_mb"] = round(peak_vram, 2)
    return result


def save_result(clip_id, result):
    output_subdir = OUTPUT_DIR / MODEL_NAME
    output_subdir.mkdir(parents=True, exist_ok=True)
    filename = f"{MODEL_ID:05d}_{clip_id:05d}.json"
    output_path = output_subdir / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return output_path


def main():
    print("\n开始 Eagle2.5-8B 推理")

    clips = load_clip_manifest()
    total_clips = len(clips)
    print(f"视频片段: {total_clips}")

    # 加载模型
    model_fn = load_model()

    completed = 0
    skipped = 0

    for clip_info in clips:
        clip_id = clip_info['id']

        # 跳过已存在的
        output_path = OUTPUT_DIR / MODEL_NAME / f"{MODEL_ID:05d}_{clip_id:05d}.json"
        if output_path.exists():
            skipped += 1
            if skipped <= 5 or skipped % 50 == 0:
                print(f"  [{clip_id}/{total_clips}] 跳过（已存在）")
            continue

        print(f"  [{clip_id}/{total_clips}] 处理: {clip_info.get('filename', '')}")
        result = run_inference_for_clip(model_fn, clip_info)

        if result is not None:
            save_result(clip_id, result)
            completed += 1

    print(f"\n[OK] Eagle2.5-8B 推理完成:")
    print(f"  完成: {completed}/{total_clips}")
    print(f"  跳过: {skipped}/{total_clips}")

    # 写完成标记
    completion_marker = OUTPUT_DIR / MODEL_NAME / "_COMPLETED.json"
    completion_info = {
        "model_name": MODEL_NAME,
        "model_id": MODEL_ID,
        "completed": completed,
        "skipped": skipped,
        "total_clips": total_clips,
        "finished_at": datetime.now().isoformat(),
        "status": "COMPLETED"
    }
    with open(completion_marker, 'w', encoding='utf-8') as f:
        json.dump(completion_info, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
