# Kaggle GPU Studio

本仓库是 [TizenryA/colab-gpu-studio](https://github.com/TizenryA/colab-gpu-studio) 的 Kaggle 迁移版，保留原项目的移动端界面、ComfyUI 工作流、任务队列和审查流程。

- **GPU**：任务提交为私有 Kaggle Notebook，默认请求 T4；可用性与免费配额由 Kaggle 决定。
- **模型**：一次性上传私有 Kaggle Dataset。执行时从只读挂载目录校验并链接到 ComfyUI，缺文件直接报错，不从 Google Drive/Hugging Face 自动重新下载模型。
- **素材与结果**：原始上传素材和生成结果存 Google Drive，同时保留本地副本用于网页预览。Drive 凭据只留在 Studio 服务端。
- **默认安全状态**：真实 GPU 仍默认关闭；离线 Mock 不使用 GPU。旧 Colab Provider 保留用于兼容，已有部署不会被自动切换。

**首次使用请阅读 [Kaggle 部署与迁移说明](docs/KAGGLE_SETUP.md)。** 原项目完整说明保留在 [README.colab.md](README.colab.md)。

## 快速开始

建议部署在 Linux VPS 或 Docker 的 Linux 容器中。后端沿用上游 Linux 文件锁，原生 Windows 不能代替 Linux 服务端。

```bash
git clone https://github.com/qinfengge/colab-gpu-studio.git
cd colab-gpu-studio
uv sync --extra test
uv run python scripts/init_env.py --site-url https://你的域名
cp docker-compose.example.yml docker-compose.yml
docker compose up -d --build
```

先在网页设置中绑定 Google Drive，再按部署说明设置 Kaggle 用户名、API 凭据、私有 Dataset 和 Drive 账号 ID。配置文件不会提交到 GitHub。最后启用 `GPU_REAL_ENABLED=1`、重启服务，在工作流页完成真实冒烟，再选择「Kaggle GPU」生成。

## 与原项目的区别

Kaggle 使用批次 Notebook，每个任务独立启动；不支持原项目的实时 Comfy 工作台和保留实例复用。取消请求后，在 Kaggle 确认任务结束前会保留清理状态；也可在 Kaggle 网页手动停止 Notebook。不会把本地取消误报为远端 GPU 已释放。

模型挂载不等于零启动成本：哈希校验、ComfyUI/节点安装和模型加载仍需要时间。两块 T4 的显存不会自动合并；原来依赖 L4/大内存的工作流必须重新验证。当前结果传输总量上限为每任务 100 MB。

## 验证与实际运行边界

```bash
uv run pytest -q
uv run ruff check api tests/gpu/test_kaggle.py scripts/prepare_kaggle_dataset.py
npm --prefix web ci
npm --prefix web run test:contract
npm --prefix web run build
```

[GitHub Actions](https://github.com/qinfengge/colab-gpu-studio/actions/workflows/kaggle.yml) 在 Linux 执行后端回归、迁移专项测试、前端契约检查和构建。自动测试使用本地替身，不会访问真实 Kaggle/Drive、上传模型或消耗 GPU。真实平台分配、私有 Dataset 挂载速度和具体模型出图仍需在用户账号上验证。

保留上游作者署名及 **AGPL-3.0** 许可证。第三方模型和节点仍遵守各自许可证。
