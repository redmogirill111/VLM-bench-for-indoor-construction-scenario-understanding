"""
Prompt 模板定义 — VLM 横向对比评测
"""

# T1: 全局描述
T1_PROMPT = "请详细描述这段视频的内容，包括人员、设备、材料、施工阶段和环境特征。"

# T2: 安全合规
T2_PROMPT = "根据建筑安全规范，指出视频中可能存在的安全隐患或违规行为（如未戴安全帽、材料杂乱、电线裸露、无警示标识等）。若无，请说明。"

# T3: 结构化输出
T3_PROMPT = "以严格 JSON 格式输出：{\"objects\": [...], \"activities\": [...], \"risks\": [...], \"phase\": \"...\"}。仅输出 JSON，无其他文本。"

# T4: 阶段推断
T4_PROMPT = "推断当前处于哪个施工工序，并说明下一步最可能的作业内容。要求给出依据。"

# T5: 负向验证
T5_PROMPT_PREFIX = "视频中是否包含以下物体？"
T5_PROMPT_SUFFIX = "请逐项回答是/否并说明理由。"

# 负向物体词库（室内施工词表中不存在的词语，用于测试幻觉抑制）
NEGATIVE_OBJECTS = [
    "游泳池", "起重机", "塔吊", "电梯", "停车场",
    "加油站", "地铁", "机场", "港口", "商场",
    "餐厅", "酒店", "银行", "医院", "学校"
]

# 温度设置
TEMPERATURE_SETTINGS = {
    "T1": 0.7,
    "T2": 0.0,
    "T3": 0.0,
    "T4": 0.7,
    "T5": 0.0
}

# 重复推理次数
REPEAT_COUNTS = {
    "T1": 3,
    "T2": 3,
    "T3": 1,
    "T4": 3,
    "T5": 1
}

def get_t5_prompt():
    """生成 T5 任务的具体 prompt（随机选择 2 个负向物体）"""
    import random
    selected_objects = random.sample(NEGATIVE_OBJECTS, 2)
    objects_str = "、".join(selected_objects)
    return f"{T5_PROMPT_PREFIX} [{objects_str}]。{T5_PROMPT_SUFFIX}"

def get_all_prompts():
    """获取所有任务的 prompt 字典"""
    return {
        "T1": T1_PROMPT,
        "T2": T2_PROMPT,
        "T3": T3_PROMPT,
        "T4": T4_PROMPT,
        "T5": get_t5_prompt()  # 每次调用生成新的负向物体组合
    }
