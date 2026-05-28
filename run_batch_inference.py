"""
VLM 横向对比评测 — 批量推理主脚本
对 9 个模型在 1,154 段视频上进行无标签推理，输出保存到 H:\benchmark\output
"""

import os
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
import sys

# 添加 scripts 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

import torch
import yaml

# 优化: 启用 PyTorch 内置 Flash SDPA kernel（比 math 实现快 2-3x）
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

from prompts import get_all_prompts, TEMPERATURE_SETTINGS, REPEAT_COUNTS, NEGATIVE_OBJECTS

# 加载模型配置
CONFIG_PATH = Path(__file__).parent / "model_configs.yaml"
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

# 数据路径配置
CLIPS_DIR = Path(CONFIG['data']['clips_dir'])
MANIFEST_FILE = Path(CONFIG['data']['manifest_file'])
OUTPUT_DIR = Path(CONFIG['data']['output_dir'])
GPU_ID = CONFIG['inference']['gpu_id']
MAX_NEW_TOKENS = CONFIG['inference']['max_new_tokens']

# 确保 ffmpeg 在 PATH 中（vlm_bench 环境）
ffmpeg_bin = r"D:\miniconda\envs\vlm_bench\Library\bin"
os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")

print("=" * 80)
print("VLM 横向对比评测 — 批量推理")
print("=" * 80)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"GPU: GPU {GPU_ID} - {torch.cuda.get_device_name(GPU_ID) if torch.cuda.is_available() else 'N/A'}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print("=" * 80)


def load_clip_manifest():
    """加载视频片段 manifest"""
    if not MANIFEST_FILE.exists():
        print(f"警告: Manifest 文件不存在: {MANIFEST_FILE}")
        # 直接从目录获取所有 mp4 文件
        clips = list(CLIPS_DIR.glob("*.mp4"))
        print(f"从目录获取到 {len(clips)} 个视频文件")
        return [{"id": i+1, "filename": clip.name} for i, clip in enumerate(sorted(clips))]
    
    with open(MANIFEST_FILE, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    # manifest 格式: {"clips": [{...}, {...}, ...], "total_clips": 1154}
    # 每个clip有: file, source, frame_start, frame_end, num_frames, duration_sec
    if isinstance(manifest, dict):
        clips = manifest.get('clips', [])
        # 添加 id 字段（从1开始）和将 file 映射为 filename
        for i, clip in enumerate(clips):
            clip['id'] = i + 1
            clip['filename'] = clip.get('file', '')
    else:
        clips = manifest
    
    print(f"[OK] 加载 manifest: {len(clips)} 个视频片段")
    return clips


def get_video_duration(video_path):
    """获取视频时长（秒）"""
    try:
        import subprocess
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        print(f"  警告: 无法获取视频时长: {e}")
        return 0.0


def model_loader_factory(model_config):
    """根据模型配置创建对应的加载器"""
    model_type = model_config['model_type']
    path = model_config['path']
    dtype = model_config['dtype']
    trust_remote_code = model_config.get('trust_remote_code', False)
    
    print(f"\n加载模型: {model_config['name']} ({model_type})")
    print(f"  路径: {path}")
    print(f"  dtype: {dtype}")
    
    device = f"cuda:{GPU_ID}"
    
    try:
        if model_type == "internvl":
            from transformers import AutoTokenizer, AutoModel
            from torchvision.transforms import InterpolationMode
            import torchvision.transforms as T
            
            IMAGENET_MEAN = (0.485, 0.456, 0.406)
            IMAGENET_STD = (0.229, 0.224, 0.225)
            
            def build_transform(input_size=448):
                return T.Compose([
                    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                    T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                ])
            
            def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
                best_ratio_diff = float('inf')
                best_ratio = (1, 1)
                area = width * height
                for ratio in target_ratios:
                    target_aspect_ratio = ratio[0] / ratio[1]
                    ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                    if ratio_diff < best_ratio_diff:
                        best_ratio_diff = ratio_diff
                        best_ratio = ratio
                    elif ratio_diff == best_ratio_diff:
                        if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                            best_ratio = ratio
                return best_ratio
            
            def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
                orig_width, orig_height = image.size
                aspect_ratio = orig_width / orig_height
                target_ratios = set(
                    (i, j) for n in range(min_num, max_num + 1)
                    for i in range(1, n + 1) for j in range(1, n + 1)
                    if i * j <= max_num and i * j >= min_num
                )
                target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
                target_aspect_ratio = find_closest_aspect_ratio(
                    aspect_ratio, target_ratios, orig_width, orig_height, image_size)
                target_width = image_size * target_aspect_ratio[0]
                target_height = image_size * target_aspect_ratio[1]
                blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
                resized_img = image.resize((target_width, target_height))
                processed_images = []
                for i in range(blocks):
                    box = (
                        (i % (target_width // image_size)) * image_size,
                        (i // (target_width // image_size)) * image_size,
                        ((i % (target_width // image_size)) + 1) * image_size,
                        ((i // (target_width // image_size)) + 1) * image_size
                    )
                    processed_images.append(resized_img.crop(box))
                if use_thumbnail and len(processed_images) != 1:
                    processed_images.append(image.resize((image_size, image_size)))
                return processed_images
            
            def internvl_load_video(video_path, num_segments=8, max_num=1):
                """Load video frames as pixel_values for InternVL"""
                from PIL import Image
                import cv2
                cap = cv2.VideoCapture(str(video_path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_indices = [int(i * total_frames / num_segments) for i in range(num_segments)]
                
                transform = build_transform(input_size=448)
                pixel_values_list = []
                num_patches_list = []
                
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(frame_rgb).convert('RGB')
                        tiles = dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=max_num)
                        pv = torch.stack([transform(t) for t in tiles])
                        num_patches_list.append(pv.shape[0])
                        pixel_values_list.append(pv)
                cap.release()
                
                if pixel_values_list:
                    pixel_values = torch.cat(pixel_values_list)
                else:
                    pixel_values = None
                return pixel_values, num_patches_list
            
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code, use_fast=False)
            model = AutoModel.from_pretrained(
                path,
                torch_dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16,
                trust_remote_code=trust_remote_code
            ).eval().to(device)
            
            def infer_fn(video_path, prompt, temperature):
                pixel_values, num_patches_list = internvl_load_video(video_path, num_segments=8, max_num=1)
                
                if pixel_values is not None:
                    pixel_values = pixel_values.to(torch.bfloat16 if dtype == "bfloat16" else torch.float16).to(device)
                    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
                    question = video_prefix + prompt
                else:
                    question = prompt
                    num_patches_list = None
                
                do_sample = temperature > 0
                gen_config = dict(max_new_tokens=MAX_NEW_TOKENS)
                if do_sample:
                    gen_config.update(temperature=temperature, do_sample=True, top_p=0.9)
                else:
                    gen_config.update(do_sample=False)
                
                with torch.no_grad():
                    response = model.chat(
                        tokenizer, pixel_values, question, gen_config,
                        num_patches_list=num_patches_list,
                        history=None, return_history=False,
                    )
                return response
            
            return infer_fn
        
        elif model_type == "videollama3":
            import importlib.util
            import types
            from transformers import AutoTokenizer, AutoModelForCausalLM

            # 设置 CUDA 显存分配策略
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).eval().to(device)

            # ---- 构建伪包以支持 processing_videollama3.py 的相对导入 ----
            pkg_name = "videollama3_pkg"
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(Path(path))]
            pkg.__file__ = str(Path(path) / "__init__.py")
            sys.modules[pkg_name] = pkg

            # 加载 image_processing 子模块
            spec_ip = importlib.util.spec_from_file_location(
                f"{pkg_name}.image_processing_videollama3",
                str(Path(path) / "image_processing_videollama3.py"),
            )
            ip_mod = importlib.util.module_from_spec(spec_ip)
            ip_mod.__package__ = pkg_name
            sys.modules[f"{pkg_name}.image_processing_videollama3"] = ip_mod
            setattr(pkg, "image_processing_videollama3", ip_mod)
            spec_ip.loader.exec_module(ip_mod)
            image_processor = ip_mod.Videollama3ImageProcessor.from_pretrained(path)

            # 加载 processing 子模块（可使用 from . import ...）
            spec_proc = importlib.util.spec_from_file_location(
                f"{pkg_name}.processing_videollama3",
                str(Path(path) / "processing_videollama3.py"),
            )
            proc_mod = importlib.util.module_from_spec(spec_proc)
            proc_mod.__package__ = pkg_name
            sys.modules[f"{pkg_name}.processing_videollama3"] = proc_mod
            setattr(pkg, "processing_videollama3", proc_mod)
            spec_proc.loader.exec_module(proc_mod)

            # 读取配置
            _proc_config = json.load(open(Path(path) / "processor_config.json", encoding='utf-8'))
            video_merge_size = _proc_config.get("video_merge_size", 2)
            image_merge_size = _proc_config.get("image_merge_size", 1)

            # 构造原生 processor（max_frames=2 防止 OOM，token compression 会进一步压缩）
            processor = proc_mod.Videollama3Qwen2Processor(
                image_processor=image_processor,
                tokenizer=tokenizer,
                video_merge_size=video_merge_size,
                image_merge_size=image_merge_size,
                fps=1,
                max_frames=2,
            )

            # 模型内置 token compression（use_token_compression=True）自动压缩相似帧

            def infer_fn(video_path, prompt, temperature):
                # 使用原生 processor 处理
                from transformers.feature_extraction_utils import BatchFeature

                # 强制中文输出
                chinese_prompt = prompt + "\n请务必使用中文回答。"

                conversation = [
                    {"role": "user", "content": [
                        {"type": "video", "video": {"video_path": str(video_path)}},
                        {"type": "text", "text": chinese_prompt},
                    ]},
                ]

                # 1. 原生加载视频（ffmpeg, fps=1, max_frames=4）
                conversation = processor._load_multimodal_data(conversation)
                # 2. 收集视觉数据
                images = processor._gather_multimodal_data(conversation)
                # 3. 处理图像（含 resize, normalize, merge）
                image_inputs = processor.process_images(images, return_tensors="pt")
                # 4. 应用 chat template（为每帧生成 <image> token）
                prompt_text = processor.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True,
                )
                # 5. 展开 <image> token（每帧的 <image> 展开为 tokens_per_frame 个）
                text_inputs = processor.process_text(
                    prompt_text, image_inputs, padding=False, padding_side="right",
                )
                # 转为 tensor（process_text 可能返回 list）
                if not isinstance(text_inputs["input_ids"], torch.Tensor):
                    text_inputs["input_ids"] = torch.tensor(text_inputs["input_ids"])
                    text_inputs["attention_mask"] = torch.tensor(text_inputs["attention_mask"])
                # 确保 2D (batch, seq_len)
                if text_inputs["input_ids"].dim() == 1:
                    text_inputs["input_ids"] = text_inputs["input_ids"].unsqueeze(0)
                    text_inputs["attention_mask"] = text_inputs["attention_mask"].unsqueeze(0)

                # 合并所有输入
                inputs = BatchFeature(data={**text_inputs, **image_inputs})
                inputs = inputs.to(device)
                # pixel_values 需要 bfloat16
                if inputs.pixel_values.dtype != torch.bfloat16:
                    inputs.data["pixel_values"] = inputs.pixel_values.to(torch.bfloat16)

                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        pixel_values=inputs.pixel_values,
                        grid_sizes=inputs.grid_sizes,
                        merge_sizes=inputs.merge_sizes,
                        modals=inputs.modals,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else None,
                    )

                # 注意：model.generate() 自定义方法返回的 output_ids 只包含生成部分
                output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                torch.cuda.empty_cache()
                return output

            return infer_fn
        
        elif model_type == "minicpmv":
            from transformers import AutoModel, AutoTokenizer
            from PIL import Image
            
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            ).eval().to(device)
            
            def infer_fn(video_path, prompt, temperature):
                # 加载多帧 PIL Image
                frames = load_video_frames(video_path, num_frames=4)
                
                # MiniCPM-V chat: content list can contain PIL Image objects directly
                content = list(frames) + [prompt]
                msgs = [{"role": "user", "content": content}]
                
                with torch.no_grad():
                    response = model.chat(
                        image=None,
                        msgs=msgs,
                        tokenizer=tokenizer,
                        sampling=True if temperature > 0 else False,
                        temperature=temperature if temperature > 0 else 0.0,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                return response
            
            return infer_fn
        
        elif model_type == "qwen3_vl":
            from transformers import AutoTokenizer, AutoProcessor
            from transformers import Qwen3VLForConditionalGeneration
            from qwen_vl_utils import process_vision_info
            
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                # 限制每帧分辨率以加速
                # attn_implementation="flash_attention_2",  # 需要安装 flash-attn
            ).eval().to(device)
            
            # torch.compile 在 generate() 场景下可能不兼容，跳过
            # 优化主要靠: Flash SDPA + 限制帧数/分辨率
            
            try:
                processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
            except Exception:
                processor = None
            
            def infer_fn(video_path, prompt, temperature):
                messages = [{"role": "user", "content": [
                    {"type": "video", "video": str(video_path),
                     "nframes": 4,
                     "resized_height": 224,
                     "resized_width": 224},
                    {"type": "text", "text": prompt}
                ]}]
                
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                if processor is not None:
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to(device)
                else:
                    inputs = tokenizer(text, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else None,
                    )
                
                output = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                return output
            
            return infer_fn
        
        elif model_type == "ovis":
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, PreTrainedModel

            # 修补 AutoConfig.register 以跳过已注册的模型类型（如 aimv2）
            _orig_register = AutoConfig.register
            def _patched_register(cls, model_type_str, config, exist_ok=True):
                return _orig_register(model_type_str, config, exist_ok=True)
            AutoConfig.register = classmethod(_patched_register)

            # 新版 transformers 移除了一些 PreTrainedModel 属性，需要补回
            if not hasattr(PreTrainedModel, 'is_parallelizable'):
                PreTrainedModel.is_parallelizable = True
            if not hasattr(PreTrainedModel, '_skip_keys_device_placement'):
                PreTrainedModel._skip_keys_device_placement = None
            if not hasattr(PreTrainedModel, '_keep_in_fp32_modules'):
                PreTrainedModel._keep_in_fp32_modules = None
            if not hasattr(PreTrainedModel, 'all_tied_weights_keys'):
                PreTrainedModel.all_tied_weights_keys = {}

            # 加载 config 并将 flash_attention_2 改为 sdpa（Windows 兼容）
            config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
            if getattr(config, 'llm_attn_implementation', None) == 'flash_attention_2':
                config.llm_attn_implementation = 'sdpa'

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    config=config,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                ).eval().to(device)
            finally:
                AutoConfig.register = _orig_register

            text_tokenizer = model.get_text_tokenizer()
            visual_tokenizer = model.get_visual_tokenizer()

            def infer_fn(video_path, prompt, temperature):
                # Ovis 使用 model.preprocess_inputs 处理图像
                from PIL import Image
                import cv2

                frames = load_video_frames(video_path, num_frames=4)
                # Ovis2 仅支持单图输入，使用第一帧
                img = frames[0] if frames else Image.new('RGB', (224, 224), (128, 128, 128))

                query = "<image>\n" + prompt
                _, input_ids, pixel_values = model.preprocess_inputs(query, [img], max_partition=9)
                attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
                input_ids = input_ids.unsqueeze(0).to(device=model.device)
                attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
                if pixel_values is not None:
                    pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
                pixel_values = [pixel_values]

                gen_kwargs = dict(
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=temperature > 0,
                    eos_token_id=model.generation_config.eos_token_id,
                    pad_token_id=text_tokenizer.pad_token_id,
                    use_cache=True,
                )
                if temperature > 0:
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["top_p"] = 0.9

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        **gen_kwargs,
                    )[0]

                output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
                return output

            return infer_fn

        elif model_type == "molmo2":
            from transformers import AutoProcessor, AutoModelForImageTextToText

            processor = AutoProcessor.from_pretrained(
                path, trust_remote_code=True
            )
            model = AutoModelForImageTextToText.from_pretrained(
                path,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            ).eval().to(device)

            def infer_fn(video_path, prompt, temperature):
                # Molmo2 official inference uses apply_chat_template
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "video", "video": str(video_path)},
                        ],
                    }
                ]

                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else None,
                    )

                generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
                output = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                return output

            return infer_fn

        elif model_type == "eagle25":
            from transformers import AutoProcessor, AutoModel

            model = AutoModel.from_pretrained(
                path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).eval().to(device)
            processor = AutoProcessor.from_pretrained(
                path, trust_remote_code=True, use_fast=True,
            )
            processor.tokenizer.padding_side = "left"

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

        elif model_type == "qwen25_vl":
            from transformers import AutoTokenizer, AutoProcessor
            from transformers import Qwen2_5_VLForConditionalGeneration
            from qwen_vl_utils import process_vision_info

            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).eval().to(device)

            try:
                processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
            except Exception:
                processor = None

            def infer_fn(video_path, prompt, temperature):
                messages = [{"role": "user", "content": [
                    {"type": "video", "video": str(video_path),
                     "nframes": 4,
                     "resized_height": 224,
                     "resized_width": 224},
                    {"type": "text", "text": prompt}
                ]}]

                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                if processor is not None:
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to(device)
                else:
                    inputs = tokenizer(text, return_tensors="pt").to(device)

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else None,
                    )

                output = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                return output

            return infer_fn

        else:
            # 通用 HF 模型加载
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
            
            try:
                processor = AutoProcessor.from_pretrained(path, trust_remote_code=trust_remote_code)
            except:
                processor = None
            
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16,
                trust_remote_code=trust_remote_code,
            ).eval().to(device)
            
            def infer_fn(video_path, prompt, temperature):
                # 多帧图像
                frames = load_video_frames(video_path, num_frames=4)
                
                if processor is not None and hasattr(processor, 'apply_chat_template'):
                    # 尝试使用 processor
                    messages = [{"role": "user", "content": []}]
                    for frame in frames:
                        messages[0]["content"].append({"type": "image", "image": frame})
                    messages[0]["content"].append({"type": "text", "text": prompt})
                    
                    inputs = processor.apply_chat_template(
                        messages,
                        images=frames,
                        return_tensors="pt",
                        add_generation_prompt=True
                    ).to(device)
                else:
                    # 回退到 tokenizer
                    text = f"<image>\n" * len(frames) + prompt
                    inputs = tokenizer(text, images=frames, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else None,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                    )
                
                output = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                return output
            
            return infer_fn
            
    except Exception as e:
        print(f"  [ERROR] 加载失败: {e}")
        traceback.print_exc()
        return None


def load_video_frames(video_path, num_frames=4):
    """从视频中均匀采样指定数量的帧"""
    import cv2
    from PIL import Image
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    cap.release()
    
    return frames


def run_inference_for_clip(model_name, model_fn, clip_info):
    """对单个视频片段进行完整推理（T1-T5）"""
    clip_file = clip_info.get("filename", "")
    clip_path = CLIPS_DIR / clip_file
    clip_id = clip_info.get("id", 0)
    
    if not clip_path.exists():
        print(f"  [ERROR] 视频文件不存在: {clip_path}")
        return None
    
    # 准备负向物体（T5 使用）
    import random
    negative_objects = random.sample(NEGATIVE_OBJECTS, 2)
    
    result = {
        "model_name": model_name,
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
            # 为 T5 生成特定的负向物体 prompt
            import random
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
                vram_before = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                
                output = model_fn(clip_path, prompt, temperature)
                
                torch.cuda.synchronize()
                latency_ms = (time.time() - t0) * 1000
                vram_after = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                
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
                result["tasks"][task_name]["runs"].append({
                    "run_id": run_id,
                    "output": f"ERROR: {str(e)}",
                    "latency_ms": 0,
                    "vram_mb": 0
                })
    
    result["total_latency_ms"] = round(total_latency, 2)
    result["peak_vram_mb"] = round(peak_vram, 2)
    
    return result


def save_result(model_name, model_id, clip_id, result):
    """保存推理结果到 JSON 文件"""
    output_subdir = OUTPUT_DIR / model_name
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{model_id:05d}_{clip_id:05d}.json"
    output_path = output_subdir / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return output_path


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("开始批量推理")
    print("=" * 80)
    
    # 加载视频清单
    clips = load_clip_manifest()
    total_clips = len(clips)
    print(f"\n待处理视频片段: {total_clips}")
    
    # 获取模型列表
    models = CONFIG['models']
    total_models = len(models)
    print(f"待推理模型: {total_models}")
    
    total_inferences = total_clips * total_models
    print(f"总推理任务数: {total_inferences}")
    
    # 遍历每个模型
    for model_config in models:
        model_id = model_config['id']
        model_name = model_config['name']
        
        print(f"\n{'='*80}")
        print(f"模型 {model_id}/{total_models}: {model_name}")
        print(f"{'='*80}")
        
        # 检查是否被手动跳过
        if model_config.get('skip', False):
            print(f"  [SKIP] 模型 {model_name} 已标记跳过")
            continue
        
        # 检查是否已全部完成，跳过加载
        completion_marker = OUTPUT_DIR / model_name / "_COMPLETED.json"
        if completion_marker.exists():
            print(f"  [SKIP] 模型 {model_name} 已完成，跳过加载")
            continue
        
        # 检查是否所有 clip 都已有结果（无 _COMPLETED 但全部存在）
        model_output_dir = OUTPUT_DIR / model_name
        existing_results = len(list(model_output_dir.glob("*.json"))) if model_output_dir.exists() else 0
        if existing_results >= total_clips:
            print(f"  [SKIP] 模型 {model_name} 所有结果已存在 ({existing_results}/{total_clips})，跳过加载")
            # 补写完成标记
            completion_info = {
                "model_name": model_name,
                "model_id": model_id,
                "completed": 0,
                "skipped": total_clips,
                "total_clips": total_clips,
                "finished_at": datetime.now().isoformat(),
                "status": "COMPLETED"
            }
            with open(completion_marker, 'w', encoding='utf-8') as f:
                json.dump(completion_info, f, ensure_ascii=False, indent=2)
            continue
        
        # 检查模型路径是否存在且包含必要文件
        model_path = Path(model_config['path'])
        if not model_path.exists() or not (model_path / "config.json").exists():
            print(f"  [SKIP] 模型 {model_name} 路径无效或未下载完成，跳过")
            continue
        
        # 加载模型
        model_fn = model_loader_factory(model_config)
        if model_fn is None:
            print(f"[ERROR] 模型 {model_name} 加载失败，跳过")
            continue
        
        # 遍历每个视频片段
        completed = 0
        skipped = 0
        
        for clip_info in clips:
            clip_id = clip_info['id']
            
            # 检查是否已存在结果
            output_path = OUTPUT_DIR / model_name / f"{model_id:05d}_{clip_id:05d}.json"
            if output_path.exists():
                skipped += 1
                if skipped <= 5 or (skipped % 50 == 0):
                    print(f"  [{clip_id}/{total_clips}] 跳过（已存在）")
                continue
            
            # 执行推理
            print(f"  [{clip_id}/{total_clips}] 处理: {clip_info.get('filename', '')}")
            result = run_inference_for_clip(model_name, model_fn, clip_info)
            
            if result is not None:
                save_result(model_name, model_id, clip_id, result)
                completed += 1
        
        print(f"\n[OK] 模型 {model_name} 完成:")
        print(f"  完成: {completed}/{total_clips}")
        print(f"  跳过: {skipped}/{total_clips}")
        
        # 写入模型完成标记文件（供看门狗和 agent 检测）
        completion_marker = OUTPUT_DIR / model_name / "_COMPLETED.json"
        completion_info = {
            "model_name": model_name,
            "model_id": model_id,
            "completed": completed,
            "skipped": skipped,
            "total_clips": total_clips,
            "finished_at": datetime.now().isoformat(),
            "status": "COMPLETED"
        }
        with open(completion_marker, 'w', encoding='utf-8') as f:
            json.dump(completion_info, f, ensure_ascii=False, indent=2)
        print(f"  完成标记已写入: {completion_marker}")
        
        # 卸载模型释放显存
        del model_fn
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 80)
    print("批量推理完成！")
    print("=" * 80)
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 统计输出文件
    total_outputs = sum(len(list((OUTPUT_DIR / m['name']).glob("*.json"))) for m in models)
    print(f"生成的输出文件: {total_outputs}/{total_inferences}")


if __name__ == "__main__":
    main()
