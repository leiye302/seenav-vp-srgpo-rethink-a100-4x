# SeeNav Route-SFT：复审版 v2

使用 [SeeNav RETHINK 独立仓库](https://github.com/leiye302/seenav-vp-srgpo-rethink-a100-4x) 的默认分支。
完整图像与数据文件位于 [本仓库的 GitHub Release](https://github.com/leiye302/seenav-vp-srgpo-rethink-a100-4x/releases/tag/seenav-route-sft-reviewed-v2)；
不需要原采集压缩包或发布者的工作目录。代码附带每份文件的 SHA-256。

```bash
git clone https://github.com/leiye302/seenav-vp-srgpo-rethink-a100-4x.git
cd seenav-vp-srgpo-rethink-a100-4x
python download_dataset.py --output data/seenav-reviewed-v2
python rethink.py fetch
python train_route_sft.py check --dataset data/seenav-reviewed-v2
```

下载器使用 Python 3.10+ 标准库，可续接未完成的下载；`--metadata-only` 仅取小体积元数据，不能训练。
完整数据与锁定基模的训练命令见 [METHOD.md](METHOD.md)。

## 数据及划分

候选来源是最终复审后的 3,764 条 Good 和 2,429 条 Bad。训练与内部验证按场景划分，
检查场景、轨迹、图像和可见模型输入不交叉。每批固定 5 Good / 3 Bad，按场景、动作及来源分层取样，
以固定种子打乱，不重复凑数。未用于固定配比的候选不放进发布包。

| 划分 | Good | Bad | 总数 | 批次 |
| --- | ---: | ---: | ---: | ---: |
| 训练 | 3,680 | 2,208 | 5,888 | 736 |
| 内部验证 | 80 | 48 | 128 | 16 |

文件大小与 SHA-256 见 [dataset_release.json](dataset_release.json)。
内部验证是 teacher-forced loss；不与作者的固定 60 题评测混用。对固定评测列表核对了源数据的精确初始状态与决策状态，
不以此宣称训练和评测的场景名称完全不相交。

- Good：保留模型自然采样的动作，不把合理探索仅因为没立刻成功而改判。
- Bad：使用已复审的单步纠正目标 `<RETHINK><action>n</action>`，未确认的纠正不放入训练。
- 输入包含任务与可见历史 $H$、双视图图像和原生成推理 $R$；仅后缀及 EOS 用于监督。图片是 AI2-THOR 渲染的
  1000×500 RGB PNG（左俯视、右第一视角），不是现实环境照片。分片内只含引用的图像；不含原始整轨迹、审核记录、
  隐藏位姿或权重。

## 来源差异

`source_lane` 区分两类证据：从原采样记录取 $H,R$ 与图像的样本，和使用重放时复审过的可见状态/图像、保留原 $R$ 的样本。
对于后者，保持被审核的状态与目标配对，只将任务句标点对齐原生训练模板，不能将其描述为模型在重放帧上重新生成的 reasoning。
原采样历史保留采集时实际可见的内容，未补入未来反馈；与作者单独评测入口显示的历史不保证逐字一致。
归档判断和重放证据都不表示新近又跑了一遍模拟器，也不保证该纠正动作在任意状态都能成功。

环境、视图与动作协议沿用 [SeeNav-Agent](https://github.com/WzcTHU/SeeNav-Agent) /
[EmbodiedBench](https://github.com/EmbodiedBench/EmbodiedBench)，仿真来自 [AI2-THOR](https://github.com/allenai/ai2thor)。
该包是衍生的 Route-SFT 训练数据，不是上述项目的官方数据发布。
