# T5b / T5c 幻觉词库 中英文对照表（50 + 50）

> 来源：`H:\benchmark\scripts\t5v2_library.py`（T5B_LIB / T5C_LIB）

> 用途：T5v2 双题幻觉测试。T5b 探针问"视频中是否存在该物体"；T5c 探针问"画面中是否存在该属性/状态/工序"。每个 clip 固定种子各抽 1 题，9 模型同题。


## 一、T5b 存在性幻觉库（50 词）

限定原则：室内施工画面必不可能出现的施工物体（室外/大尺度专属设备与结构）。


### 土方与路面机械 Earthwork & Pavement Machinery

| # | 中文 | English |
|---|------|---------|
| 1 | 塔式起重机（塔吊） | Tower crane |
| 2 | 履带式挖掘机 | Crawler excavator |
| 3 | 轮式装载机 | Wheel loader |
| 4 | 推土机 | Bulldozer |
| 5 | 平地机 | Motor grader |
| 6 | 压路机 | Road roller |
| 7 | 沥青摊铺机 | Asphalt paver |
| 8 | 沥青洒布车 | Asphalt distributor truck |

### 起重与垂直运输 Lifting & Vertical Transport

| # | 中文 | English |
|---|------|---------|
| 9 | 汽车起重机 | Truck crane |
| 10 | 履带起重机 | Crawler crane |
| 11 | 轮胎起重机 | Rubber-tyred crane |
| 12 | 缆索起重机 | Cable crane |
| 13 | 龙门起重机 | Gantry crane |
| 14 | 桅杆式起重机 | Derrick crane |
| 15 | 施工升降机（外用施工电梯） | Construction hoist (external elevator) |
| 16 | 井字物料提升机 | Material hoist (mast-climbing) |
| 17 | 混凝土泵车 | Concrete pump truck |
| 18 | 混凝土搅拌运输车 | Concrete mixer truck |
| 19 | 自卸卡车 | Dump truck |

### 桩工与地基处理 Piling & Ground Improvement

| # | 中文 | English |
|---|------|---------|
| 20 | 旋挖钻机 | Rotary drilling rig |
| 21 | 振动沉管打桩机 | Vibro pile driver |
| 22 | 静压桩机 | Static pile press |
| 23 | 强夯机 | Dynamic compaction machine |
| 24 | 锚杆钻机 | Anchor drilling rig |

### 桥梁与隧道机械 Bridge & Tunnel Machinery

| # | 中文 | English |
|---|------|---------|
| 25 | 架桥机 | Bridge girder erection machine |
| 26 | 悬浇挂篮 | Balanced cantilever formwork (form traveler) |
| 27 | 移动模架 | Movable formwork (MSS) |
| 28 | 盾构机 | Shield tunneling machine |
| 29 | 悬臂掘进机 | Roadheader |
| 30 | 多臂凿岩台车 | Multi-boom drilling jumbo |
| 31 | 隧道衬砌台车 | Tunnel lining trolley |
| 32 | 湿喷机械手 | Wet-shotcrete manipulator |
| 33 | 顶管机 | Pipe jacking machine |

### 搅拌站与砂石处理 Mixing Plants & Aggregate

| # | 中文 | English |
|---|------|---------|
| 34 | 稳定土拌合站 | Stabilized-soil mixing plant |
| 35 | 沥青搅拌站 | Asphalt mixing plant |
| 36 | 混凝土搅拌站 | Concrete batching plant |
| 37 | 砂石分离机 | Aggregate separator |

### 铁路与索道 Railway & Ropeway

| # | 中文 | English |
|---|------|---------|
| 38 | 铺轨机 | Track-laying machine |
| 39 | 铁路捣固车 | Railway tamping machine |
| 40 | 接触网作业车 | Catenary maintenance vehicle |
| 41 | 架空客运索道 | Aerial passenger ropeway |

### 水工、港工与船舶 Hydraulic, Port & Marine

| # | 中文 | English |
|---|------|---------|
| 42 | 集装箱岸桥 | Container quay crane (STS) |
| 43 | 打桩船 | Piling barge |
| 44 | 挖泥船 | Dredger |
| 45 | 浮吊船 | Floating crane ship |
| 46 | 水电站弧形闸门 | Hydropower radial gate |
| 47 | 船闸人字门 | Ship lock miter gate |

### 大尺度室外结构 Large Outdoor Structures

| # | 中文 | English |
|---|------|---------|
| 48 | 风力发电机组 | Wind turbine generator |
| 49 | 双曲线冷却塔 | Hyperbolic cooling tower |
| 50 | 工业烟囱 | Industrial chimney |

## 二、T5c 属性幻觉库（50 词）

限定原则：室内施工画面必不可能出现的物体属性 / 场景状态 / 不相容工序（室外天气、大尺度空间、物理错配、室外专属工序、人员装备错配）。


### 室外天气与气候 Outdoor Weather & Climate

| # | 中文 | English |
|---|------|---------|
| 1 | 被积雪覆盖的脚手架 | Scaffolding covered in snow |
| 2 | 被大雪覆盖的材料堆 | Material piles buried in snow |
| 3 | 结满冰凌的临时电线 | Temporary wires with hanging icicles |
| 4 | 被暴雨淋透的作业面 | Work surface drenched by rainstorm |
| 5 | 遭遇冰雹袭击的楼板 | Floor slab struck by hailstorm |
| 6 | 沙尘暴笼罩的作业区 | Work zone shrouded in a sandstorm |
| 7 | 被洪水淹没的走廊 | Corridor flooded by water |
| 8 | 被大风吹翻的安全网 | Safety net torn by strong wind |
| 9 | 被阳光暴晒开裂的模板 | Formwork cracked by sun exposure |
| 10 | 被雨水冲出沟痕的地面 | Ground gullied by rainwater erosion |

### 大尺度空间属性 Large-Scale Spatial Attributes

| # | 中文 | English |
|---|------|---------|
| 11 | 直接可见的天空与云朵 | Directly visible sky and clouds |
| 12 | 繁星满天的夜空 | Night sky full of stars |
| 13 | 悬挂在百米高空的作业平台 | Work platform suspended hundreds of meters high |
| 14 | 正在坍塌的基坑边坡 | Collapsing foundation pit slope |
| 15 | 低空掠过画面上空的飞机 | Aircraft flying low over the scene |
| 16 | 横跨材料堆放区的彩虹 | Rainbow spanning the material yard |
| 17 | 紧邻悬崖的作业面 | Work surface beside a cliff |
| 18 | 紧邻大海的作业面 | Work surface next to the sea |
| 19 | 沙漠环抱的施工作业区 | Construction site surrounded by desert |
| 20 | 火山口旁的施工作业区 | Construction site beside a volcanic crater |

### 材料物理属性错配 Physically Impossible Material Attributes

| # | 中文 | English |
|---|------|---------|
| 21 | 金色镜面材质的脚手架 | Golden mirror-finished scaffolding |
| 22 | 透明玻璃制成的混凝土墙 | Wall built of transparent glass-concrete |
| 23 | 橡胶质地的钢筋 | Rubber-made rebar |
| 24 | 融化成液体的钢梁 | Steel beam melted into liquid |
| 25 | 用冰块砌筑的墙体 | Wall built of ice blocks |
| 26 | 燃烧中的模板支撑架 | Burning formwork support frame |
| 27 | 悬浮在半空中的材料箱 | Material case floating in mid-air |
| 28 | 爬满藤蔓植物的墙面 | Wall covered in climbing vines |
| 29 | 覆盖海藻的模板 | Formwork covered in seaweed |
| 30 | 被岩浆包裹的电梯井 | Elevator shaft enveloped in lava |
| 31 | 用雪堆成的砂石堆 | Sand pile made of snow |
| 32 | 铺满玫瑰花瓣的作业面 | Work surface covered with rose petals |

### 室外专属/不相容工序 Outdoor-Only or Incompatible Operations

| # | 中文 | English |
|---|------|---------|
| 33 | 正在进行的路面沥青摊铺作业 | Ongoing asphalt pavement paving operation |
| 34 | 正在进行的桥梁合龙施工 | Ongoing bridge closure (joining) construction |
| 35 | 正在进行的隧道爆破作业 | Ongoing tunnel blasting operation |
| 36 | 正在进行的山体爆破采石作业 | Ongoing mountain quarry blasting |
| 37 | 正在进行的屋面防水热熔施工 | Ongoing roof waterproofing torch-applied works |
| 38 | 正在进行的外墙玻璃幕墙吊装 | Ongoing curtain-wall glazing hoisting |
| 39 | 正在施划的道路标线 | Road markings being painted |
| 40 | 正在进行的铁路换轨作业 | Ongoing railway rail-replacement operation |
| 41 | 正在进行的船体除锈喷涂作业 | Ongoing ship-hull derusting and painting |
| 42 | 塔吊正在画面中吊运材料 | Tower crane lifting materials on screen |
| 43 | 直升机正在画面中吊运材料 | Helicopter lifting materials on screen |
| 44 | 消防云梯车正在画面中展开救援 | Fire ladder truck deployed for rescue |

### 人员与装备错配 Personnel & Equipment Mismatch

| # | 中文 | English |
|---|------|---------|
| 45 | 佩戴潜水装备的作业人员 | Worker wearing diving gear |
| 46 | 身穿宇航服的作业人员 | Worker wearing a spacesuit |
| 47 | 骑马搬运材料的工人 | Worker transporting materials on horseback |
| 48 | 驾驶雪地摩托的工人 | Worker riding a snowmobile |
| 49 | 牵着骆驼运输材料的队伍 | Caravan of camels carrying materials |
| 50 | 乘坐热气球作业的工人 | Worker operating from a hot-air balloon |