# VLM Construction-Site Benchmark

Benchmark dataset and inference scripts for evaluating Vision-Language Models (VLMs) on construction site video understanding.

## Dataset

- **1154 video clips** split from 8 raw videos of hospital construction sites
- Scenes include: indoor/outdoor construction, structural work, MEP, safety hazards
- Clip format: MP4, ~0.1–5 MB each, ~10 seconds per clip
- Source videos located in `video_clips/`

### Video Sources

| Video | Description | Duration |
|-------|-------------|----------|
| Hospital construction site | Main site walkthrough | ~2h |
| rgb-1 ~ rgb-7 | Robot front camera feeds | 30s–5min |

## Inference Scripts

Located in `inference/`:

| Script | Models Covered | Description |
|--------|---------------|-------------|
| `run_batch_inference.py` | All 9 models | **Main entry point** — unified batch runner with per-model adapters |
| `run_eagle25.py` | Eagle2.5-8B | Standalone Eagle2.5 inference |
| `run_videollama3_inference.py` | VideoLLaMA3-7B | Standalone VideoLLaMA3 inference |
| `prompts.py` | — | T1–T5 task prompt definitions |
| `model_configs.yaml` | — | Model paths and configurations |

### Supported Models

| # | Model | Type | Input Mode |
|---|-------|------|------------|
| 1 | VideoLLaMA3-7B | VLM | Native video |
| 2 | Qwen2.5-VL-7B-Instruct | VLM | Native video |
| 3 | Qwen3-VL-8B-Instruct | VLM | Native video |
| 4 | InternVL2.5-8B | VLM | Multi-frame images |
| 5 | InternVL3-8B | VLM | Multi-frame images |
| 6 | MiniCPM-V-2.6-8B | VLM | Native video |
| 7 | Ovis2-8B | VLM | Frame extraction |
| 8 | Eagle2.5-8B | VLM | Native video |
| 9 | Molmo2-8B | VLM | Native video |

### Tasks (T1–T5)

| Task | Description | Output |
|------|-------------|--------|
| T1 | Scene Description | Free-text |
| T2 | Safety Hazard Detection | Free-text |
| T3 | Structured JSON Output | JSON |
| T4 | Construction Phase Inference | Free-text |
| T5 | Hallucination / Negative Object Test | Free-text |

## Tools

- `tools/split_videos.py` — Split raw videos into short clips (~10s each)

## Usage

```bash
# Run all models on all clips
python inference/run_batch_inference.py

# Run a specific model
python inference/run_eagle25.py
python inference/run_videollama3_inference.py
```

## Requirements

- Python 3.10+
- PyTorch 2.x + CUDA
- transformers >= 4.45
- See individual scripts for model-specific dependencies
