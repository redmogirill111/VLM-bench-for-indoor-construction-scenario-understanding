# -*- coding: utf-8 -*-
"""
T5v2 幻觉测试词库与判分工具（回应审稿意见 4.1 的重新设计）

设计原则：
1. T5B_LIB 存在性幻觉库（50 词）：室内施工画面必不可能出现的施工物体
   —— 全部为室外/大尺度专属设备与结构（土方、起重、桩工、路面、桥梁、
      隧道、水工、港工、铁路、风电等），与 62 词室内施工领域词表不相交；
2. T5C_LIB 属性幻觉库（50 词）：室内施工画面必不可能出现的物体属性 /
   场景状态 / 不相容工序（室外天气、大尺度空间、物理错配属性、
      室外专属工序、人员装备错配）；
3. 每 clip 由固定种子确定性抽取 1 个 T5B 词 + 1 个 T5C 词，
   同一 clip 对全部 9 个模型呈现完全相同的两道题（跨模型公平可比、可复现）；
4. 两道题的正确答案恒为"否"（由构造先验保证，无需人工真值即可自动判分）；
5. M4v2 判分：两项均正确拒绝（都答"否"）记 1，其余（任一答"是"或解析失败）记 0。
"""

SEED = 20260818

# ── T5B：存在性幻觉库（50）────────────────────────────────────────
T5B_LIB = [
    # 土方与路面机械（室外专属）
    "塔式起重机（塔吊）",
    "履带式挖掘机",
    "轮式装载机",
    "推土机",
    "平地机",
    "压路机",
    "沥青摊铺机",
    "沥青洒布车",
    # 起重与垂直运输（安装在结构外围，室内画面不可能出现）
    "汽车起重机",
    "履带起重机",
    "轮胎起重机",
    "缆索起重机",
    "龙门起重机",
    "桅杆式起重机",
    "施工升降机（外用施工电梯）",
    "井字物料提升机",
    "混凝土泵车",
    "混凝土搅拌运输车",
    "自卸卡车",
    # 桩工与地基处理
    "旋挖钻机",
    "振动沉管打桩机",
    "静压桩机",
    "强夯机",
    "锚杆钻机",
    # 桥梁与隧道工程机械
    "架桥机",
    "悬浇挂篮",
    "移动模架",
    "盾构机",
    "悬臂掘进机",
    "多臂凿岩台车",
    "隧道衬砌台车",
    "湿喷机械手",
    "顶管机",
    # 搅拌站与砂石处理
    "稳定土拌合站",
    "沥青搅拌站",
    "混凝土搅拌站",
    "砂石分离机",
    # 铁路与索道
    "铺轨机",
    "铁路捣固车",
    "接触网作业车",
    "架空客运索道",
    # 水工、港工与船舶
    "集装箱岸桥",
    "打桩船",
    "挖泥船",
    "浮吊船",
    "水电站弧形闸门",
    "船闸人字门",
    # 大尺度室外结构
    "风力发电机组",
    "双曲线冷却塔",
    "工业烟囱",
]

# ── T5C：属性幻觉库（50）──────────────────────────────────────────
T5C_LIB = [
    # 室外天气与气候属性（室内无开放大气）
    "被积雪覆盖的脚手架",
    "被大雪覆盖的材料堆",
    "结满冰凌的临时电线",
    "被暴雨淋透的作业面",
    "遭遇冰雹袭击的楼板",
    "沙尘暴笼罩的作业区",
    "被洪水淹没的走廊",
    "被大风吹翻的安全网",
    "被阳光暴晒开裂的模板",
    "被雨水冲出沟痕的地面",
    # 大尺度空间属性（室内不可见的天外景观）
    "直接可见的天空与云朵",
    "繁星满天的夜空",
    "悬挂在百米高空的作业平台",
    "正在坍塌的基坑边坡",
    "低空掠过画面上空的飞机",
    "横跨材料堆放区的彩虹",
    "紧邻悬崖的作业面",
    "紧邻大海的作业面",
    "沙漠环抱的施工作业区",
    "火山口旁的施工作业区",
    # 材料物理属性错配（构造上不可能）
    "金色镜面材质的脚手架",
    "透明玻璃制成的混凝土墙",
    "橡胶质地的钢筋",
    "融化成液体的钢梁",
    "用冰块砌筑的墙体",
    "燃烧中的模板支撑架",
    "悬浮在半空中的材料箱",
    "爬满藤蔓植物的墙面",
    "覆盖海藻的模板",
    "被岩浆包裹的电梯井",
    "用雪堆成的砂石堆",
    "铺满玫瑰花瓣的作业面",
    # 室外专属 / 与室内阶段不相容的工序
    "正在进行的路面沥青摊铺作业",
    "正在进行的桥梁合龙施工",
    "正在进行的隧道爆破作业",
    "正在进行的山体爆破采石作业",
    "正在进行的屋面防水热熔施工",
    "正在进行的外墙玻璃幕墙吊装",
    "正在施划的道路标线",
    "正在进行的铁路换轨作业",
    "正在进行的船体除锈喷涂作业",
    "塔吊正在画面中吊运材料",
    "直升机正在画面中吊运材料",
    "消防云梯车正在画面中展开救援",
    # 人员与装备错配
    "佩戴潜水装备的作业人员",
    "身穿宇航服的作业人员",
    "骑马搬运材料的工人",
    "驾驶雪地摩托的工人",
    "牵着骆驼运输材料的队伍",
    "乘坐热气球作业的工人",
]

assert len(T5B_LIB) == 50, f"T5B_LIB 应为 50 词，实际 {len(T5B_LIB)}"
assert len(T5C_LIB) == 50, f"T5C_LIB 应为 50 词，实际 {len(T5C_LIB)}"


# ── 确定性抽样与 Prompt 构造 ─────────────────────────────────────
def sample_items(clip_id):
    """对给定 clip_id 确定性抽取 (t5b_item, t5c_item)。
    同一 clip_id 在任何进程、任何模型下抽取结果一致。"""
    import random
    rng = random.Random((int(clip_id) * 1000003) ^ SEED)
    return rng.choice(T5B_LIB), rng.choice(T5C_LIB)


def build_prompt(clip_id):
    """构造 T5v2 prompt：单次推理包含两道必然为假的判断题。"""
    item_b, item_c = sample_items(clip_id)
    prompt = (
        "请仅根据视频画面回答以下两个问题：\n"
        f"1. 视频画面中是否存在“{item_b}”？\n"
        f"2. 视频画面中是否存在“{item_c}”？\n"
        "请分别回答“是”或“否”，并简要说明理由。"
    )
    return prompt, item_b, item_c


# ── 回答解析与判分 ────────────────────────────────────────────────
# 否定/肯定模式（正则）：解析时取“出现位置最早”的模式类别；
# 正则排除子串误命中：“是否”中的“否/是”、“没有/所有”中的“有”、“但是”中的“是”等。
import re as _re

NEG_RES = [_re.compile(p) for p in [
    "不存在", "并未出现", "没有出现", "未出现", "并未", "没有",
    "不是", "并无", "均不", "都不", "看不见", "看不到", "(?<!是)否",
    # 英文模式（Molmo2 等英文输出模型）
    r"\b[Nn]o\b", "not present", "not visible", "cannot be seen", "can't be seen",
    "do not see", "don't see", "does not see", "does not contain", "doesn't contain",
    "there is no", "there isn't", "no sign of", "no evidence", "not shown",
    "cannot identify", "can't identify", "unable to see", "absent", "n't appear",
    "not any", "negative",
]]
POS_RES = [_re.compile(p) for p in [
    "(?<!是否)存在", "(?<!是否)出现", "(?<![没所拥具罕稀])有", "(?<![但也还于就乃是])是(?!否)",
    # 英文模式
    r"\b[Yy]es\b", "is present", "are present", "can be seen", "appears",
    "there is a", "there are", "is visible", "is shown", "contains", "positive",
]]


def _earliest(text, regexes):
    best_pos = None
    for rx in regexes:
        m = rx.search(text)
        if m and (best_pos is None or m.start() < best_pos):
            best_pos = m.start()
    return best_pos


def _classify_segment(seg):
    """对一段文本判定 yes/no/unparsed。
    若段内含问号（模型复述问题），仅取最后一个问号之后的文本判定。"""
    if not seg or not seg.strip():
        return "unparsed"
    q = max(seg.rfind("？"), seg.rfind("?"))
    if q != -1:
        seg = seg[q + 1:]
        if not seg.strip():
            return "unparsed"
    pn = _earliest(seg, NEG_RES)
    pp = _earliest(seg, POS_RES)
    if pn is None and pp is None:
        return "unparsed"
    if pn is not None and (pp is None or pn <= pp):
        return "no"
    return "yes"


def parse_answers(text):
    """解析模型回答 → (ans1, ans2)，取值 yes / no / unparsed。

    策略 A：按题号标记（1. / 1、 / 问题1 / 第1 …）切两段分别判定；
    策略 B：无题号时，若出现“都/均/两者/两项”总述则整体判定两题同值；
    策略 C：否则取前两个独立判定词依序对应两题；再失败则均记 unparsed。
    """
    if not text:
        return "unparsed", "unparsed"

    import re
    t = text.replace("\r", "\n")

    # ── 策略 A：题号分段 ──
    m = re.search(r"(?:^|\n|\s|。|；|;|\))\s*(?:问题\s*|第\s*)?1\s*[\.、．:：）)]", t)
    n = re.search(r"(?:^|\n|\s|。|；|;|\))\s*(?:问题\s*|第\s*)?2\s*[\.、．:：）)]", t)
    if m and n and n.end() > m.end():
        seg1 = t[m.end(): n.start()]
        seg2 = t[n.end():]
        a1, a2 = _classify_segment(seg1), _classify_segment(seg2)
        if "unparsed" not in (a1, a2):
            return a1, a2
        if a1 != "unparsed" and a2 == "unparsed":
            # 第 2 题答案可能紧跟在第 1 题理由后，无显式题号
            rest = t[n.end():]
            a2 = _classify_segment(rest)
            if a2 != "unparsed":
                return a1, a2
        if a1 == "unparsed" and a2 != "unparsed":
            pre = t[: m.start()]
            a1 = _classify_segment(pre)
            if a1 != "unparsed":
                return a1, a2
        # 落入策略 B/C 兜底
    elif m and not n:
        seg1 = t[m.end():]
        a1 = _classify_segment(seg1)
        if a1 != "unparsed":
            # 第 2 题可能在后文无编号作答
            return a1, "unparsed"

    # ── 策略 B：总述（都/均/两者/两项）──
    head = t[:200]
    pn = _earliest(head, NEG_RES)
    pp = _earliest(head, POS_RES)
    if any(k in head for k in ("都", "均", "两者", "两项", "全部")):
        if pn is not None and (pp is None or pn <= pp):
            return "no", "no"
        if pp is not None:
            return "yes", "yes"

    # ── 策略 C：依序取前两个判定词（否定短语区间屏蔽其中的肯定子串）──
    neg_spans = []
    hits = []
    for rx in NEG_RES:
        for m in rx.finditer(t):
            neg_spans.append((m.start(), m.end()))
            hits.append((m.start(), "no"))
    def _covered(i, spans):
        return any(s <= i < e for s, e in spans)
    for rx in POS_RES:
        for m in rx.finditer(t):
            if not _covered(m.start(), neg_spans):
                hits.append((m.start(), "yes"))
    hits.sort()
    if len(hits) >= 2:
        return hits[0][1], hits[1][1]
    if len(hits) == 1:
        return hits[0][1], "unparsed"

    return "unparsed", "unparsed"


def score_m4(ans1, ans2):
    """M4v2 判分：两项均正确拒绝（no）记 1，否则 0。"""
    return 1 if (ans1 == "no" and ans2 == "no") else 0


# ── 自测 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) 抽样确定性
    for cid in (1, 419, 447, 1154):
        b, c = sample_items(cid)
        b2, c2 = sample_items(cid)
        assert (b, c) == (b2, c2), f"clip {cid} 抽样不确定"
        print(f"clip {cid}: T5B={b} | T5C={c}")

    # 2) 全库无重复、两库不相交
    assert len(set(T5B_LIB)) == 50 and len(set(T5C_LIB)) == 50
    assert not (set(T5B_LIB) & set(T5C_LIB)), "两库存在重叠"

    # 3) 解析用例
    cases = [
        ("1. 否，画面中为室内走廊，不存在塔式起重机。2. 否，室内没有冰雪。", ("no", "no")),
        ("1、是，画面中似乎有挖掘机。2、否。", ("yes", "no")),
        ("问题1：视频中并不存在该物体。问题2：也没有出现。", ("no", "no")),
        ("两项均不存在。", ("no", "no")),
        ("第1题答：存在。第2题答：不存在。", ("yes", "no")),
        ("1. 不存在。", ("no", "unparsed")),
        ("画面中没有看到相关内容。", ("no", "unparsed")),
        ("这个视频展示了室内施工场景。", ("unparsed", "unparsed")),
        # 刁钻用例：是否/所有/但是 等子串陷阱
        ("1. 画面中是否存在塔吊？不存在。2. 是否有积雪？没有。", ("no", "no")),
        ("所有的物体都看不见。", ("no", "no")),
        ("第1项：是。第2项：但是画面较暗，无法确认，倾向不存在。", ("yes", "no")),
        # 英文输出用例（Molmo2 等）
        ("1. No, the video does not contain a tower crane. 2. No, there is no snow.", ("no", "no")),
        ("1. Yes, I can see a paver in the frame. 2. No.", ("yes", "no")),
        ("1. The object is not present in the video. 2. It is not visible either.", ("no", "no")),
        ("1. Yes. 2. Yes, both are visible.", ("yes", "yes")),
    ]
    for text, expect in cases:
        got = parse_answers(text)
        status = "OK " if got == expect else "FAIL"
        print(f"[{status}] {text[:30]}... -> {got} (期望 {expect})")

    print("\n词库与解析自测完成。")
