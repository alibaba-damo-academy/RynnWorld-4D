# RynnWorld4D Policy

基于 RynnWorld4D (3-branch Wan2.2) backbone + Flow Matching 的机器人操作策略。

## 代码来源

| 项目 | 说明 |
|------|------|
| **本仓库（rynnworld4d_policy）** | 本目录，VPP + RynnWorld4D backbone 精简版 |
| **原版 VPP 代码** | <https://github.com/roboterax/video-prediction-policy> |
| **RynnWorld4D 代码** | 本仓库根目录（`../`） |

### 论文

- **VPP (Video Prediction Policy)**: [arXiv 2412.14803](https://arxiv.org/abs/2412.14803)
- **RynnWorld4D**: 3-branch Wan2.2 (video + depth + optical flow) 世界模型

## 相对于原版 VPP 的改动

### 1. Backbone: SVD -> RynnWorld4D (3-branch Wan2.2)

**原版**：使用 Stable Video Diffusion (SVD) 的 UNet 提取视觉特征，condition_dim = 1280。

**改动**：替换为 RynnWorld4D 的 3-branch Wan2.2 Transformer，三个分支分别建模 video / depth / optical flow，特征沿 channel 维度拼接。

- `wan_feature_extractor.py`：封装了 Wan2.2 VAE 编码 + Transformer 单步前向 + hook 提取中间层特征
- RynnWorld4D 模式下，hook 捕获 `(video, depth, flow)` 三路输出，concat 后 condition_dim = 3 x 3072 = **9216**
- 加载 RynnWorld4D Stage-1 SFT checkpoint，训练时 backbone 全部冻结

对应文件：
- `policy_models/module/wan_feature_extractor.py` — 新增（原版无此文件）

### 2. Policy Head: EDM Diffusion -> Flow Matching

**原版**：使用 EDM (Karras) score matching + DDIM/Heun 等多步采样器，训练和推理步骤复杂。

**改动**：替换为 Conditional Flow Matching，大幅简化：

- **训练**：采样 t ~ U(0,1)，线性插值 x_t = (1-t)*noise + t*data，预测速度场 v = data - noise，MSE loss
- **推理**：4 步 Euler ODE 从噪声积分到数据，比 EDM 的 10 步 DDIM 更快更简洁

对应文件：
- `policy_models/edm_diffusion/flow_matching.py` — 新增

### 3. 输入增加本体感知 (Proprioception)

**原版**：policy head 只接收视觉特征 + 语言目标。

**改动**：增加当前机器人关节状态 `observation.state`（54 维）作为额外输入：

- DiffusionTransformer 的 `proprio_emb` MLP 将 54 维状态映射到 384 维 token
- 拼入 encoder context：`[goal(1), state_images(224), proprio(1)]` = 226 tokens
- 注意：使用的是 `observation.state`（传感器反馈的实际关节位置），不是 `action`（发给电机的控制指令）

### 4. 输入分辨率: 480x640 -> 224x224

**原版**：使用 SVD 的默认分辨率。

**改动**：输入 RGB 改为 224x224，数据增强为 Resize(256) + RandomCrop(224) + ColorJitter。训练速度提升约 3 倍。

### 5. 数据集: Calvin -> Tianji

**原版**：使用 Calvin 仿真数据集。

**改动**：新增 Tianji 天机双臂机器人数据集支持：

- 读取 episode 目录下的 mp4 视频 + parquet 动作/状态文件
- action_dim = 54（7 arm_L + 7 arm_R + 20 hand_L + 20 hand_R）
- 支持 action 归一化（per-dim mean/std）

对应文件：
- `policy_models/datasets/tianji_dataset.py` — 新增

### 6. 代码精简

移除了原版中不需要的组件：
- SVD backbone（StableVideoDiffusionPipeline, Diffusion_feature_extractor）
- EDM 相关代码（GCDenoiser, gc_sampling, 各种 noise schedule/sampler）
- Gripper 双相机逻辑
- Calvin/XBot 数据集和评估脚本
- Stage-1 视频训练代码

## 架构

```
Head Camera RGB (224x224)
    |
    v  [frozen] RynnWorld4D Backbone (Wan2.2 5B, 3-branch)
    |    |- Video branch  -> (B, F*H*W, 3072)
    |    |- Depth branch  -> (B, F*H*W, 3072)
    |    |- Flow branch   -> (B, F*H*W, 3072)
    |    -> concat channel -> (B, F_tok, 9216, H_tok, W_tok)
    |
    v  [trainable] Video_Former (Perceiver Resampler, 3D)
    |    -> (B, 224, 384)
    |
    v  [trainable] Flow Matching Policy Head (DiffusionTransformer)
    |    Encoder context: [goal(1), state_images(224), proprio(1)]
    |    Decoder: cross-attention -> predict velocity field
    |    Training: MSE(v_pred, v_target)
    |    Inference: 4-step Euler ODE
    |
    -> Predicted actions (B, 10, 54)  # 未来10步, 54维关节动作
```

## 模型权重

| 权重 | 路径 | 说明 |
|------|------|------|
| Wan2.2-TI2V-5B | `./pretrained/Wan2.2-TI2V-5B-Diffusers` | 基础 Wan 模型，HuggingFace 下载 |
| RynnWorld4D SFT | `<repo>/training/.../checkpoint-NNN` | Stage-2 训练产出（见 `scripts/rynnworld4d-stage2.sh`） |
| CLIP ViT-B/32 | 自动下载 | 语言目标编码 |

## 训练数据

仓库内 `data/tianji_sample/` 已附 3 条示例 episode（head 视频 + parquet + metadata），可直接训练。完整 Tianji-Wuji Pick-Place 数据集结构：

```
<root_data_dir>/Pick-Place/
├── episode_000000/
│   ├── observation.images.head.mp4       # head 相机视频
│   ├── timeseries.parquet                # action(54d) + observation.state(54d)
│   └── metadata.json                     # task_prompt, fps
├── episode_000001/
...
└── episode_000249/                       # 共 250 episodes
```

## 运行

```bash
cd rynnworld4d_policy

# 单卡训练
python train.py

# 指定数据路径
python train.py --root_data_dir /path/to/data

# 指定模型路径
python train.py --wan_model_path /path/to/Wan2.2-TI2V-5B-Diffusers
```

## 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| backbone | rynnworld4d | 3-branch Wan2.2 |
| policy_type | flow_matching | 4-step Euler ODE |
| action_dim | 54 | 双臂 + 双灵巧手 |
| proprio_dim | 54 | observation.state |
| num_latents | 224 | Video_Former query 数 |
| condition_dim | 9216 | 3 x 3072 (3-branch) |
| lr | 1e-4 | AdamW |
| batch_size | 1 | 受 GPU 显存限制（~43GB） |
| resolution | 224x224 | Resize(256) + RandomCrop(224) |

## 文件说明

```
train.py                                  # 训练入口
policy_conf/
  train_config.yaml                       # 训练配置
  datamodule/tianji.yaml                  # 数据集配置
policy_models/
  vpp_policy.py                           # 主模型（VPP_Policy）
  edm_diffusion/
    flow_matching.py                      # Flow Matching policy head
  module/
    wan_feature_extractor.py              # RynnWorld4D backbone 特征提取
    Video_Former.py                       # Perceiver Resampler (3D)
    diffusion_decoder.py                  # DiffusionTransformer (encoder-decoder)
    clip_lang_encoder.py                  # CLIP 语言编码器
    clip.py                               # CLIP 模型实现
    transformers/                         # Transformer blocks, position embeddings
  datasets/
    tianji_dataset.py                     # 天机数据集 + DataModule
  utils/
    lr_schedulers/                        # Tri-stage LR scheduler
    clip_tokenizer.py                     # CLIP tokenizer
```
