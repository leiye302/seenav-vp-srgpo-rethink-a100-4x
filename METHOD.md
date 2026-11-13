# SeeNav RETHINK：原生 VP + SRGPO + OPSD 上的两阶段适配

基于 [SeeNav VP + SRGPO + OPSD 四卡基线](https://github.com/leiye302/seenav-vp-srgpo-opsd-a100-4x)。
保留 SeeNav 原生双视图、最近五步动作历史、0–7 离散动作、AI2-THOR 执行与奖励、SRGPO 优势和原始 OPSD 损失。
唯一必须换的是输出接口：基线的 JSON 不包含可训练的原生 `</think>`–`<action>` 决策边界。
方法版使用作者源码中已有的 VP 标签提示和动作执行器，不另写任务模板或改变观察。

| 实验组 | Actor 起点 | 输出 | Teacher |
| --- | --- | --- | --- |
| 原基线 `reproduce.py` | Qwen2.5-VL-3B-Instruct | 原生 JSON | 原静态 skill |
| `tag-control` | 同一基模 | 原生 tags | 静态 skill，只将 JSON 的末句换成标签指令 |
| `sft-static` | Route-SFT | 原生 tags | 同一标签静态 skill |
| `rethink` | 同一 Route-SFT | 原生 tags | 静态 skill；在 Actor 自主触发的位置加局部 hint |

想量化局部 hint 的单独贡献，用 `sft-static` 对比 `rethink`；不要用 JSON 基线的差值单独归因给 hint。
这三个标签组的模拟器、验证任务、采样参数和训练配方相同。保持原生动作解析和错误处理；标签之间的
软连字符 U+00AD 可忽略，正常换行可保留，不做动作重写或强制解码。

## 一键串行训练

先在 Linux 服务器上检出方法分支。按 [README 的环境说明](README.md#准备环境)
创建 Python 3.10 环境，运行官方安装脚本并安装原生依赖。配置使用四张已分配的 A100 80 GB；
Route-SFT 只用其中第一张，进程退出后，第二阶段才用四张。

```bash
git clone --branch rethink-method https://github.com/leiye302/seenav-vp-srgpo-opsd-a100-4x.git
cd seenav-vp-srgpo-opsd-a100-4x
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install 'setuptools==75.1.0' wheel
python rethink.py fetch
```

完成 README 中的依赖安装后，在同一环境继续：

```bash
python download_dataset.py --output data/seenav-reviewed-v2
BASE_SNAPSHOT="$(python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    "Qwen/Qwen2.5-VL-3B-Instruct", revision="66285546d2b821cf421d4f5eb2576359d3770cd3",
    allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors"], token=False,
))
PY
)"

python run_pipeline.py check --model "$BASE_SNAPSHOT"

export SEENAV_HINT_API_URL="https://provider.example/v1/chat/completions"
export SEENAV_HINT_MODEL="your-vision-model"
read -rsp 'Hint API key: ' SEENAV_HINT_API_KEY; echo
export SEENAV_HINT_API_KEY
# 可选：不使用联网的 W&B 日志。
export WANDB_MODE=offline
python run_pipeline.py train --model "$BASE_SNAPSHOT" --gpus 0,1,2,3 \
  --output runs/seenav-rethink
```

默认读取本仓库 Release 的复审版数据。可以用 `--dataset` 指定它在别处的完整本地副本。
`check` 只读校验源码、数据和可选的模型，不占 GPU、不发 API 请求。
`train` 先做四卡和 hint 配置预检，再从基模跑 736 步 Route-SFT；
只有成功导出 `stage1/global_step_736` 且数据、步数、模型文件和 marker 校验通过，才启动原生 150 个 SRGPO 更新。
这不加载 SFT 的优化器状态。两个训练阶段也可按下方命令分别运行。

输出目录中，`stage1.log` 是 SFT 进度，`handoff.json` 记录待交接的父断点与数据指纹，
`stage2/binding.json` 记录实际验收后的父断点；`stage2/train.log` 和 `stage2/metrics/` 是二阶段训练日志，
`stage2.log` 则记录启动和交接错误。
只有最终 FSDP 断点、Hugging Face 导出和逐步指标齐全才写 `complete.json`。其中的
`hint_coverage` 是触发后实际应用 hint 的比例；`local_hint_status` 区分有应用、全部回退、完全没有触发。
如果全程没有应用任何 hint，会明确提示，
不能把这种结果当作有局部 hint 的训练。启动前请为基模、SFT 导出和原生二阶段 FSDP 断点留出空间。
如果 SFT 已完成但二阶段启动失败，不必重训 SFT；修复后，用下面的独立二阶段命令
指向原来的 `stage1/global_step_736`，并指定新的二阶段输出目录。

## 第一阶段：Route-SFT

- Actor 输入是原生 VP 提示、当前 BEV+FV 双视图、已有动作与反馈，以及该步实际采集的 `<think>…</think>` token。
  Reasoning 不重新编码、不算 CE，也不加 reasoning KL。训练不会把标签或纠错理由拼进提示。
- Good 保留**原先自然采样动作文本**，目标是 `<action>n</action>` 和 EOS。
  Bad 必须有明确、未自行修正、会误导决策的 reasoning 错误；目标是一个专用
  `<RETHINK>` token、已验收的下一步 `<action>n</action>` 和 EOS。证据不足不作为 Bad。
- 不覆盖旧模型词表，不把 marker 拆成普通字串。独立二分类损失取首个监督 token **之前**的 causal logits，
  对比 marker 与其余词表，而不是仅与 `<action>` 的首 token 二选一。
- 每个全局批次 5 Good / 3 Bad，一轮不重复；划分不共享场景、轨迹、观察图和输入指纹。
  默认是全参数 AdamW，LR `2e-6`，warmup 60，cosine，梯度裁剪 1.0，seed 0，
  FP32 权重、BF16 autocast、micro batch 1。内部验证是 teacher-forced loss，不是环境成功率。

设第 $i$ 条的前缀为 $s_i$，目标后缀为 $u_i$，路由标签为 $b_i$，$q_i$ 是首个后缀位置上 marker 的全词表概率：

$$
\mathcal{L}_{\mathrm{SFT}} =
\frac{1}{B}\sum_{i=1}^{B}\left[
-\frac{1}{|u_i|}\sum_{t=1}^{|u_i|}\log p_\theta(u_{i,t}\mid s_i,u_{i,<t})
+0.1\,\operatorname{BCE}(b_i,q_i)\right].
$$

在基线 README 的 Linux 环境中，沿用锁定上游的 Transformers 4.51.1。另需：

1. 已审核并封存的 `seenav_native_route_sft_v1` 数据包：`manifest.json`、`train.jsonl`、
   `validation.jsonl`、两个 `*_batches.jsonl`、manifest 所列的 PNG 分片。
   完整流程使用[复审版 GitHub Release](DATASET.md)，可用下方命令从仓库下载。原 605 批旧包仅为单阶段兼容输入，
   不随此次发布分发；不会自动选用未验收的候选或旧权重。
   旧包的结构化图像渲染器省略了图像前一个换行；适配器校验原摘要后，用在线 SRGPO 实际保留换行的
   原生 chat 渲染训练。原图、reasoning token、动作与监督位置不因此改写。
2. Qwen2.5-VL-3B-Instruct 的 Hub 快照 `66285546d2b821cf421d4f5eb2576359d3770cd3`。
   如果从缓存复制到自定义目录，用 `--model-receipt` 提供该快照的模型 ID、revision 与文件 SHA-256 收据。
3. 一张可用 80 GB BF16 GPU。Stage 1 使用单卡，不占用其余三张；Stage 2 使用四张。

```bash
python download_dataset.py --output data/seenav-reviewed-v2
export SFT_DATA="$PWD/data/seenav-reviewed-v2"
python rethink.py fetch
python train_route_sft.py check --dataset "$SFT_DATA" --model "$BASE_SNAPSHOT"
CUDA_VISIBLE_DEVICES=0 python train_route_sft.py train \
  --dataset "$SFT_DATA" --model "$BASE_SNAPSHOT" --output runs/seenav-stage1
```

可在 `check` 加 `--verify-frames` 完整解码并校验图片。训练只在显式 `train` 时启动。产物
`global_step_N` 是带 tokenizer、processor 和 SHA manifest 的完整 Hugging Face 模型。
`--save-training-state` 可同时归档最后一步的 optimizer、scheduler 和 RNG；Stage 2 不加载这些状态。

## 第二阶段：Actor 路由的局部 OPSD

1. Student 仍自由生成 `<think>…</think><action>n</action>`，或者在二者之间生成一个
   `<RETHINK>`。动作由原 SeeNav 执行，**所有轨迹、所有有效响应仍参与 SRGPO**。
2. 轨迹结束后，从响应 mask 内的**真实 token ID**找 marker。只有在完整 reasoning 与单个合法动作之间的 marker
   算触发，不从解码日志、Analyzer 的评分、任务失败或动作优劣推断。隐藏特殊 token 的执行解码不影响路由。
3. 只为触发步请求视觉 Analyzer。请求按轨迹合并，长轨迹按少量触发图分组；只发送触发步的当前双视图，
   配合其可见提示、reasoning 和本轨迹实际反馈。Analyzer 只建议一个当前状态可执行的下一步，
   **不判断 Good/Bad，不负责路由，也不替换实际执行动作**。详见 [hint 提示词](seenav_rethink/local_hint_system.txt)。
4. Teacher 是同一个当前 Actor 的无梯度评分视角。保留原图像张量、原响应 token、mask 和位置结构，
   只在原 Teacher 静态 skill 后附上该步提示。没触发、提示缺失/不合法、API 超时或失败、提示超长时，
   该步继续使用静态 Teacher。没有 episode Analyzer。训练时 Student 和环境不接收 hint，
   验证、部署也不访问 Analyzer。
5. 与原基线使用**同一份 SDAR loss**：`sdar_gate_beta=5`、系数 `0.01`、原响应 token-mean。
   原 reference KL、奖励、采样、SRGPO 轨迹/步骤优势、四卡配置、验证间隔和保存计划不改。
   原生 Actor/reference 仍由共享模型路径初始化；以 SFT 起跑的两组使用同一个 SFT 父断点。
   原生训练内 8 任务验证不等同于作者单独的 60 题 benchmark；本入口不把它报告为后者。

```bash
python rethink.py check --stage1 "$ROUTE_SFT_CHECKPOINT"
python rethink.py plan --variant rethink

# 使用自己的支持图像输入、OpenAI-compatible Chat Completions 的提供方。
export SEENAV_HINT_API_URL="https://provider.example/v1/chat/completions"
export SEENAV_HINT_MODEL="your-vision-model"
read -rsp 'Hint API key: ' SEENAV_HINT_API_KEY; echo
export SEENAV_HINT_API_KEY
python rethink.py train --gpus 0,1,2,3 --stage1 "$ROUTE_SFT_CHECKPOINT" --output runs/seenav-rethink
```

没有预设提供方或模型。只有 `rethink` 训练组会调用 API；请使用有权限把当前导航图和轨迹反馈发送给该服务的数据。
`tag-control` 不传 `--stage1`，`sft-static` 传同一个 `--stage1`；两者都不需要 API。
默认每请求最多 4 张图、并发 4、单次 45 秒，最多两次尝试；每个 update 的提示队列共享 180 秒预算。
可用 `SEENAV_HINT_{IMAGES_PER_REQUEST,PARALLEL,TIMEOUT,ATTEMPTS,MAX_TOKENS,UPDATE_BUDGET}` 显式配置。
训练指标中分开记录原始 marker 行、合规触发、实际应用、去重后提示数和失败回退。
没有触发或 API 持续异常时，局部提示覆盖会下降；训练不会因单个提示不可用而中止。

## 本地验收

`python -m unittest discover -s tests -v` 验证原基线不变、CE/BCE 位置、真实 token 路由、
双视图输入、Teacher mask/位置、API 分组和 fallback，以及两阶段交接和最后产物检查。测试用模拟 API，不会计费。
`fetch/check/plan` 只验证锁定源码、资产和启动配置，不代表 GPU、AI2-THOR 或真实 API 已跑通。
