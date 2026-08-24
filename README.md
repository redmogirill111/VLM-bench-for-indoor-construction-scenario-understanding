# VLM Construction-Site Benchmark

Benchmark dataset and inference scripts for evaluating Vision-Language Models (VLMs) on construction site video understanding.

## Dataset

- **1154 video clips** split from 2 construction sites (hospital/university dormitory)
- Scenes include: indoor construction, unstructural work, MEP, works, safety hazards, building materials
- Clip format: MP4 
- Source videos located in `video_clips/`

### Video Sources

| Video | Description | Duration |
|-------|-------------|----------|
| Hospital construction site (Scenario A) | Robot front camera feeds |  2,056s |
| rgb-1 ~ rgb-7 (Scenario B) | Robot front camera feeds |  5,442s |

## Inference Scripts

Located in `inference/`:

| Script | Models Covered | Description |
|--------|---------------|-------------|
| `run_batch_inference.py` | All 9 models | **Main entry point** — unified batch runner with per-model adapters (T1–T5 v1) |
| `run_eagle25.py` | Eagle2.5-8B | Standalone Eagle2.5 inference |
| `run_videollama3_inference.py` | VideoLLaMA3-7B | Standalone VideoLLaMA3 inference |
| `t5v2_library.py` | — | **T5v2** dual-probe phrase libraries (50+50), answer parser, M4v2 scoring |
| `run_t5v2_inference.py` | All 9 models | **T5v2** hallucination re-test runner (deterministic per-clip sampling) |
| `run_t5v2_all.ps1` | All 9 models | Multi-environment orchestrator for the T5v2 re-test |
| `build_t5v2_datafiles.py` | — | Rebuild evaluation data files from T5v2 outputs (M4→M4v2) |
| `prompts.py` | — | T1–T5 task prompt definitions (v1 T5 kept for reproducibility) |
| `model_configs.yaml` | — | Model paths and configurations |

### Supported Models

| # | Model | Type | Input Mode |
|---|-------|------|------------|
| 1 | VideoLLaMA3-7B | VLM | Native video |
| 2 | Qwen2.5-VL-7B-Instruct | VLM | Native video |
| 3 | Qwen3-VL-8B-Instruct | VLM | Native video |
| 4 | InternVL2.5-8B | VLM | Sampling frames (4 frames) |
| 5 | InternVL3-8B | VLM | Sampling frames (4 frames) |
| 6 | MiniCPM-V-2.6-8B | VLM | Native video |
| 7 | Ovis2-8B | VLM | Sampling frames (4 frames) |
| 8 | Eagle2.5-8B | VLM | Native video |
| 9 | Molmo2-8B | VLM | Native video |

### Model Download

All models are available on both **HuggingFace** and **ModelScope** (国内镜像).

#### HuggingFace

```bash
pip install huggingface_hub

# 1. InternVL2.5-8B
huggingface-cli download OpenGVLab/InternVL2_5-8B --local-dir ./models/InternVL2.5-8B

# 2. InternVL3-8B
huggingface-cli download OpenGVLab/InternVL3-8B --local-dir ./models/InternVL3-8B

# 3. Molmo2-8B
huggingface-cli download allenai/Molmo2-8B --local-dir ./models/Molmo2-8B

# 4. MiniCPM-V-2.6-8B
huggingface-cli download openbmb/MiniCPM-V-2_6 --local-dir ./models/MiniCPM-V-2.6-8B

# 5. Qwen3-VL-8B-Instruct
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir ./models/Qwen3-VL-8B-Instruct

# 6. VideoLLaMA3-7B
huggingface-cli download DAMO-NLP-SG/VideoLLaMA3-7B --local-dir ./models/VideoLLaMA3-7B

# 7. Ovis2-8B
huggingface-cli download AIDC-AI/Ovis2-8B --local-dir ./models/Ovis2-8B

# 8. Eagle2.5-8B
huggingface-cli download nvidia/Eagle2.5-8B --local-dir ./models/Eagle2.5-8B

# 9. Qwen2.5-VL-7B-Instruct
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./models/Qwen2.5-VL-7B-Instruct
```

Or download all at once:

```bash
for repo in OpenGVLab/InternVL2_5-8B OpenGVLab/InternVL3-8B allenai/Molmo2-8B \
  openbmb/MiniCPM-V-2_6 Qwen/Qwen3-VL-8B-Instruct DAMO-NLP-SG/VideoLLaMA3-7B \
  AIDC-AI/Ovis2-8B nvidia/Eagle2.5-8B Qwen/Qwen2.5-VL-7B-Instruct; do
  name=$(echo $repo | cut -d'/' -f2)
  huggingface-cli download $repo --local-dir ./models/$name
done
```

#### ModelScope（国内推荐）

```bash
pip install modelscope

# 1. InternVL2.5-8B
modelscope download --model OpenGVLab/InternVL2_5-8B --local_dir ./models/InternVL2.5-8B

# 2. InternVL3-8B
modelscope download --model OpenGVLab/InternVL3-8B --local_dir ./models/InternVL3-8B

# 3. Molmo2-8B
modelscope download --model allenai/Molmo2-8B --local_dir ./models/Molmo2-8B

# 4. MiniCPM-V-2.6-8B
modelscope download --model openbmb/MiniCPM-V-2_6 --local_dir ./models/MiniCPM-V-2.6-8B

# 5. Qwen3-VL-8B-Instruct
modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir ./models/Qwen3-VL-8B-Instruct

# 6. VideoLLaMA3-7B
modelscope download --model DAMO-NLP-SG/VideoLLaMA3-7B --local_dir ./models/VideoLLaMA3-7B

# 7. Ovis2-8B
modelscope download --model AIDC-AI/Ovis2-8B --local_dir ./models/Ovis2-8B

# 8. Eagle2.5-8B
modelscope download --model nvidia/Eagle2.5-8B --local_dir ./models/Eagle2.5-8B

# 9. Qwen2.5-VL-7B-Instruct
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct --local_dir ./models/Qwen2.5-VL-7B-Instruct
```

Or download all at once:

```bash
for repo in OpenGVLab/InternVL2_5-8B OpenGVLab/InternVL3-8B allenai/Molmo2-8B \
  openbmb/MiniCPM-V-2_6 Qwen/Qwen3-VL-8B-Instruct DAMO-NLP-SG/VideoLLaMA3-7B \
  AIDC-AI/Ovis2-8B nvidia/Eagle2.5-8B Qwen/Qwen2.5-VL-7B-Instruct; do
  name=$(echo $repo | cut -d'/' -f2)
  modelscope download --model $repo --local_dir ./models/$name
done
```

### Tasks (T1–T5)

| Task | Description | Output |
|------|-------------|--------|
| T1 | Scene Description | Free-text |
| T2 | Safety Hazard Detection | Free-text |
| T3 | Structured JSON Output | JSON |
| T4 | Construction Phase Inference | Free-text |
| T5 | Hallucination / Negative Object Test (v1, superseded by **T5v2** below) | Free-text |

### T5v2 — Dual-Probe Hallucination Test (revised)

The original T5 (2 randomly sampled fabricated objects per clip) discriminated weakly among models and probed only object-existence hallucination. **T5v2** redesigns the negative-verification task with two curated phrase libraries (see `t5v2_library.py` and `T5_phrase_libraries_CN_EN.md` for the full bilingual lists):

- **T5b (existence probes, 50 phrases)** — construction objects that can never appear in indoor footage (e.g., tower crane, shield tunneling machine, bridge girder erector, wind turbine);
- **T5c (attribute probes, 50 phrases)** — attributes / scene states / operations that can never occur indoors (e.g., snow-covered scaffolding, directly visible sky, asphalt paving).

Per clip, one phrase is drawn from each library with a fixed random seed keyed on `clip_id` — all nine models receive **identical** questions. The model answers two yes/no questions; the hallucination-suppression score **M4v2 = 1** only if both probes are correctly rejected ("no"), otherwise 0. The design is fully label-free and auto-scorable by construction.

```bash
# Single model
python run_t5v2_inference.py --models InternVL2.5-8B

# Multiple models
python run_t5v2_inference.py --models InternVL2.5-8B,InternVL3-8B --limit 20

# All 9 models across conda environments (Windows PowerShell)
./run_t5v2_all.ps1
```

Reference results (9 models × 1,154 clips): Qwen3-VL-8B-Instruct 0.988 > Ovis2-8B 0.971 > InternVL3-8B 0.944 > ... > MiniCPM-V-2.6-8B 0.636.

## Human-Annotation Subset (100 clips)

To validate the proxy metrics, a stratified subset of 100 clips (scene A/B × duration) was defined for human judgment. Files under `annotation/`:

| File | Description |
|------|-------------|
| `annotation/sample_100.json` / `.csv` | The 100-clip stratified sample (clip_id, file, scene, duration) |
| `annotation/annotation_guideline_CN.md` | Annotation guideline: 1–5 ordinal scores for T1 completeness/accuracy and T2 safety; binary judgments for phase correctness, term usage, existence/attribute hallucination; objective clip-level anchors |
| `annotation/annotation_template.csv` | Blank annotation template |
| `annotation/annotations_sim_v3.json` | **Pre-release placeholder** (simulated v3 annotations used for pipeline validation and correlation-matrix calibration). Will be replaced by the released human annotations |

> Graded scoring (1–5) is adopted because open-ended VQA tasks admit no single definitive reference answer — responses within a range of granularity/emphasis all count as partially correct, which binary labels would collapse.

## Tools

- `tools/split_videos.py` — Split raw videos into short clips

## Usage

```bash
# Run all models on all clips
python inference/run_batch_inference.py

# Run a specific model
python inference/run_eagle25.py
python inference/run_videollama3_inference.py

# T5v2 hallucination re-test (revised T5)
python run_t5v2_inference.py
```

## Requirements

- Python 3.10+
- PyTorch 2.x + CUDA
- transformers >= 4.45
- See individual scripts for model-specific dependencies
