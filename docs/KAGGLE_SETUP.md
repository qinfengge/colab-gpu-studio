# Kaggle 部署与迁移

## 数据流

浏览器 → Studio 队列 → 私有 Kaggle Notebook → Studio 校验结果 → Google Drive + 本地预览副本。

模型独立存于私有 Kaggle Dataset；Notebook 的 `dataset_sources` 挂载模型，生成任务不携带 Google OAuth 或 Kaggle API 密钥。Notebook 源码含本次任务参数及压缩参考图，因此同样保持私有。

## 1. 启动 Studio、连接 Drive

按根目录 README 部署 Linux/Docker 服务，保持 `GPU_REAL_ENABLED=0`。按 [OAuth 配置说明](OAUTH_SETUP.md) 设置 Google OAuth，在账号目录连接 **Drive**；Kaggle 运行不要求 Colab OAuth 或 Colab CU。

登录后从 `GET /api/gpu/accounts` 获取已连接 Drive 的 `account_id`，写入 `.env` 的 `KAGGLE_DRIVE_ACCOUNT_ID`。该 ID 是 Studio 的账号 UUID，不是邮箱、Drive 文件夹 ID 或 Kaggle 用户名。使用直接拥有 Drive 授权的账号。

原始素材上传后会同步 Drive。同步失败时本地文件仍保留，作品库显示「Drive 未同步 · 重试」。生成结果继续使用原项目的 Drive 上传、校验和重试逻辑。删除本地预览不等于删除 Drive 备份。

## 2. 一次性准备模型 Dataset

先完成模型许可和工作流依赖审查，导出包含 `weight_files` 的已审查 manifest。把所需模型准备在 CPU/VPS 的目录下，目录结构与 `weight_files[].target` 一致，例如：

```text
model-files/
  checkpoints/model.safetensors
  vae/vae.safetensors
  loras/style.safetensors
```

如果模型目前在 Drive，先在 CPU/VPS 上取回一次；不要在每次 GPU 任务中执行这一步。也可以直接在 Kaggle 网页创建私有 Dataset 并上传同样的目录结构。无论来源，都需要匹配审查清单中的 SHA-256。

仓库提供准备工具：

```bash
uv run python scripts/prepare_kaggle_dataset.py \
  --manifest reviewed-manifest.json \
  --model-dir ./model-files \
  --staging-dir ./data/kaggle-model-upload \
  --dataset YOUR_KAGGLE_USERNAME/studio-models \
  --license LICENSE_ID
```

`LICENSE_ID` 必须替换为与模型再分发许可匹配的 Kaggle 许可标识。工具会先验证全部模型的哈希与大小，使用内容哈希作为上传文件名，能硬链接时不额外复制大文件。上传完成前不要修改源文件。目标目录必须为空。

配置 Kaggle CLI 凭据后，加 `--upload` 可立即创建 Dataset（始终私有，不传 `--public`）。如果准备目录已经生成，可直接执行：

```bash
uv run python -m kaggle datasets create -p ./data/kaggle-model-upload
```

确认上传和处理完成，且 Dataset 可被该账号访问、处于 Private 状态。更新模型时准备新的内容；GPU 端依然校验固定 SHA-256，因此不会静默加载不匹配的最新文件。

运行端兼容 `/kaggle/input/SLUG`、`/kaggle/input/OWNER/SLUG` 和 `/kaggle/input/datasets/OWNER/SLUG` 等已挂载布局。若模型缺失、同一模型在多个 Dataset 中产生歧义、文件大小或哈希不匹配，任务会失败；不会回退到远程模型下载。支持准备工具的哈希文件名和手动上传的 `target` 目录结构。

## 3. 配置 Kaggle

在 `.env` 中设置以下内容（示例中的值需替换）：

```dotenv
GPU_COMPUTE_PROVIDER=kaggle
GPU_STORAGE=drive
GPU_MODEL_CACHE=kaggle
GPU_REAL_ENABLED=1
GPU_MAX_RUNTIME_MINUTES=30
KAGGLE_USERNAME=YOUR_KAGGLE_USERNAME
KAGGLE_API_TOKEN=YOUR_PRIVATE_TOKEN
KAGGLE_MODEL_DATASETS=YOUR_KAGGLE_USERNAME/studio-models
KAGGLE_DRIVE_ACCOUNT_ID=YOUR_STUDIO_DRIVE_ACCOUNT_UUID
```

也可在服务端使用官方 CLI 支持的 `kaggle.json`；Docker 需要将凭据挂入服务容器，或使用上面的环境变量。不要把凭据写入 Notebook、Dataset 或 Git 仓库。多个 Dataset 用逗号分隔，Dataset slug 必须不同。

新建配置默认选择 Kaggle；已有 `.env` 不会被初始化脚本覆盖，需要手动补充配置。保留 `GPU_COMPUTE_PROVIDER=colab` 可使用旧 Colab 工作流。

重启 API、worker 和 watchdog，使三者使用同一配置及持久化数据卷：

```bash
docker compose up -d --build
```

Kaggle 必须允许当前账号使用 GPU 和 Notebook Internet；安装固定版本 ComfyUI/依赖需要联网。GPU、周配额、并发、会话长度和资源容量以 Kaggle 账号实际显示为准，代码不绕过这些限制。当前固定请求 `NvidiaTeslaT4`，不尝试收费机型或多个账号轮换。

## 4. 工作流启用与生成

在工作流页完成模板校验、模型许可、依赖审查，然后运行 Kaggle 真实冒烟、验收和发布。Google 账号在这里仅用于 Drive；无需开通 Colab。

已在 Colab 上发布的旧工作流不能直接作为 Kaggle 验证证据。迁移时创建新的草稿版本，重新完成 Kaggle 冒烟。两块 T4 不会自动提供一整块 32 GB 显存；原 L4/高内存工作流可能需要调整模型或参数。

创作页选择「Kaggle GPU」。每个任务独立启动一个私有 Notebook，完成后取回结果并保存到 Drive。内置精选 LoRA 的权重同样要先放入 Dataset；旧的 Drive 自定义 LoRA 入口不能作为 Kaggle 模型来源，需转入 Dataset 并在工作流中锁定 target 与 SHA-256。

模型无需每次从 Drive 下载，但环境安装、模型哈希检查和加载仍消耗时间。暂不迁移 Colab 的打包 Python/CUDA 环境缓存。

## 5. 取消、故障与恢复

- 官方 CLI 的 Notebook 生命周期与 Colab 不同；本版不提供 retained/smart/batch 实例复用或实时 Comfy 工作台。
- 任务有服务器提交的 Notebook 运行时限。排队和平台终止可能有延迟；不会承诺点击取消立即释放。
- 本地取消后保持清理状态，直到 Kaggle 返回 `COMPLETE`、`ERROR` 或 `CANCEL_ACKNOWLEDGED` 等终态。可进入 Kaggle 的对应私有 Notebook 手动停止。
- 私有 Notebook 名称记录在持久化数据卷 `kaggle/<runtime-id>/handle.json`。保留该卷，避免丢失运行状态。
- 提交响应丢失时，同一任务不会盲目再提交一个版本。若查询也失败，清理状态会保留；先核对 Kaggle 后台状态和凭据，勿把未知状态当作已释放。
- 结果总量限制为 100 MB/任务。结果过大时失败，不把部分结果当作成功；长视频需要降低输出大小。
- `CU=0` 代表 Kaggle 路径不按 Colab CU 计费，不代表 GPU 配额无限或已查询到剩余额度。站内时长含调度开销，不作为 Kaggle 实际用量账单。

## 验证边界与参考

CI 不持有 Kaggle/Google 凭据，不会启动真实 GPU。它验证挂载/哈希、私有性、幂等、任务恢复、取消终态、Drive 字节校验、工作流审查/发布/提交，以及原有回归测试。

仍需用用户账号实测：GPU 实际分配、联网、Dataset 挂载、ComfyUI 依赖兼容性、具体模型显存需求及最终 Drive 回传。只有通过真实冒烟的工作流才应投入使用。

官方接口参考：[Kaggle Notebook CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)、[Notebook 元数据](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)、[Kaggle Notebooks](https://www.kaggle.com/docs/notebooks)。
