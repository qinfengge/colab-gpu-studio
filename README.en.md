> This fork adds Kaggle batch GPU execution, private Kaggle Dataset model mounts, and Google Drive input/output storage. See [Kaggle setup](docs/KAGGLE_SETUP.md) and the [current README](README.md). The upstream Colab instructions below describe the optional legacy provider. Real Kaggle/Drive execution has not yet been verified with user credentials.

**English** | [简体中文](README.md)

> Current version: 0.2.10. Fixes new-account authorization failing after OAuth application setup. Also includes Colab CLI compute accounts, independent Drive account binding, and CU reconciliation. See the [0.2.10 release notes](docs/RELEASE_NOTES_0.2.10.md) and [offline implementation notes](docs/OFFLINE_IMPLEMENTATION.md).

# Colab GPU Studio

A personal AI workspace that lets you run ComfyUI on Colab GPUs from your phone, for image generation and H3 video.

The frontend is a React single-page app optimized for mobile touch controls, and the backend uses FastAPI. Your website, task queue, and past creations normally stay on your own VPS, without occupying a GPU. Only when you click “Generate” does the backend request a GPU from Google Colab. Once the job is done, it saves the results back and **immediately destroys the runtime by default (or keeps it briefly when Retain mode is selected), so your compute units (CU) are not wasted on an idle GPU**.

## Downloads and Online Demo

- **Latest version**: [0.2.10 release](https://github.com/TizenryA/colab-gpu-studio/releases/tag/v0.2.10) (source + release notes).
- **Prebuilt multi-platform bundles**: the [0.2.10 release assets](https://github.com/TizenryA/colab-gpu-studio/releases/tag/v0.2.10) ship Windows x86_64 / Linux x86_64 / macOS x64 / macOS arm64 / Android Termux packages. Desktop bundles include Python and dependencies; Termux installs its native dependencies online at first launch; checksums are in `SHA256SUMS.txt`.
- **Read-only UI demo**: [Open the Cloudflare Workers demo](https://gpu-colab-studio-demo.tizenry.tech). Public demo password: `demo-readonly`.
- The demo uses the **original 0.1.31-dev.8 frontend** with a separate Mock data snapshot. It does not connect to the production site, Google, or a real GPU, and does not consume CU. Generation, uploads, edits, deletion, and authorization are unavailable; download and self-host the full version for complete functionality.
- [Demo details and more screenshots](docs/DEMO.md). The screenshots below were captured in a real browser on the live demo site; they are not design mockups.

| Mobile creation page | Settings · Light theme |
|---|---|
| <img src="docs/demo/screenshots/mobile-create.png" width="280" alt="Mobile creation page: black-and-white interface and offline Mock workflow"> | <img src="docs/demo/screenshots/mobile-settings-light.png" width="280" alt="Settings page: light theme and offline mode"> |

<details>
<summary>View the desktop screenshot</summary>

![Desktop creation page](docs/demo/screenshots/desktop-create.png)

</details>

## Why Build This?

1. **Saving CU comes first**: The ComfyUI environment and tens of gigabytes of large-model weights are cached in your Google Drive. Let an inexpensive CPU runtime handle the heavy lifting of downloading models and setting up the environment. The expensive GPU (T4 / L4) only does three things after startup: **restore the environment → sample the output → shut down and delete the instance immediately**.
2. **Comfortable image and video creation on your phone**: No more struggling with dense ComfyUI node connections in a mobile browser or watching a Colab tab to prevent disconnections. The frontend automatically renders forms from each workflow, while queued jobs keep running in the background even after you close the browser.
3. **One-click enablement in the browser**: built-in templates start as drafts; the Workflows page walks you through validation, license confirmation, weight locking, a real GPU smoke test and publishing, polling automatically and stopping at the failing step.
4. **Four refined workflow templates included**: Built-in templates cover Krea+MOSS dual-sampling image generation, MOSS text-to-image, MiniMax H3 image-to-video with first and last frames, and MiniMax H3 multi-reference image-to-video (supporting up to 6 reference images, 8-step Turbo v1.1, 1.2x latent refinement, a seamless anime VAE, and multiple SFW LoRAs for motion, camera movement, and consistent visual style).

> **Safety note**: a fresh install only enables the offline `demo-mock` workflow (no Google, no cost). The four real built-in workflows sit in the list as drafts; clicking "enable" and confirming the cost runs one real GPU smoke test (billed per minute) and publishes automatically once it passes. See [built-in workflows](docs/BUILTIN_WORKFLOWS.md).

## Built-in Workflows

| Workflow | Version | Recommended GPU | Description |
|---|---|---|---|
| `krea2-moss-dual-t2i` | 1.0.0 / 1.1.1 | T4 HighRAM | 1.0.0 is the original template; 1.1.1 chains 4 publicly downloadable film/MJ texture LoRAs (1.1.0 referenced 7 LoRAs but shipped none of the weight files, so it has been retired) |
| `moss-image-sft-t2i` | 1.0.0 | T4 | Single-image text-to-image generation with OpenMOSS |
| `minimax-h3-i2v` | 1.1.0 | L4 | Generate a short video with audio from a first frame and an optional last frame, with an 8-step acceleration LoRA |
| `minimax-h3-ref2v` | 1.4.0 | L4 | Generate video with audio from up to 6 reference images; choose 4-step, 8-step, refinement, or 20-step modes, with a seamless anime VAE |

See the [built-in workflow guide](docs/BUILTIN_WORKFLOWS.md) for details and activation steps.

## Quick Deployment

### Option 1: One-command Docker Startup (Recommended for VPS Deployment)

Install Docker, Docker Compose, and Python first (Python is used to generate the initial random secrets):

```bash
# 1. Generate your private .env configuration file (replace the domain with your own; leave it unchanged for local use)
uv run python scripts/init_env.py --site-url https://your-domain

# 2. Copy your own compose file (port/mount tweaks live here, so the git tree stays clean)
cp docker-compose.example.yml docker-compose.yml

# 3. Start the services (three containers: api / worker / watchdog)
docker compose up -d --build
```

> Only `docker-compose.example.yml` is tracked; your local `docker-compose.yml` is git-ignored, so changing ports or mounts no longer conflicts with `git pull`. `scripts/deploy.sh` creates it automatically when missing.

By default, the service listens only on the local address `127.0.0.1:8788`. For public access, use Caddy or Nginx as a reverse proxy with HTTPS. All data is stored in the `studio-data` volume. **Do not accidentally run `docker compose down -v`, which deletes the data volume.**

### Option 2: Run Locally

Requires Python 3.11+, `uv`, Node.js, and `npm`:

```bash
uv sync --extra test
uv run python scripts/init_env.py --site-url http://127.0.0.1:8788
npm --prefix web ci
npm --prefix web run build

# Terminal 1: Start the Web API
uv run --env-file .env uvicorn app.main:app --app-dir api --host 127.0.0.1 --port 8788 --no-access-log

# Terminal 2: Start the task queue worker
uv run --env-file .env python -m app.worker worker

# Terminal 3: Start the timeout-cleanup watchdog
uv run --env-file .env python -m app.worker watchdog
```

After startup, open `http://127.0.0.1:8788`. The login password is the `STUDIO_PASSWORD` value in the newly generated `.env` file.

## How Do I Connect My Own Google Colab Account to Run Real Models?

To prevent secret leaks and unintended charges, real GPU access is disabled by default (`GPU_REAL_ENABLED=0`). Follow these steps in order to connect your own account:

1. **Connect Google OAuth**: Follow the [Google OAuth and Drive setup guide](docs/OAUTH_SETUP.md), import your Google Cloud Web client JSON in the web settings, and connect Colab and Drive separately.
2. **Enable built-in workflows or import new ones**: Refer to the [built-in workflow guide](docs/BUILTIN_WORKFLOWS.md) and the [model and workflow review process](docs/WORKFLOW_REVIEW.md). Set `GPU_REAL_ENABLED=1`, configure budget limits, and review and enable the workflows.

## API Documentation

See the [Chinese API guide](docs/API调用文档.md) for authentication, asset uploads, jobs and batches, polling, downloads, and retries, with a [Python client example](docs/examples/studio_api_example.py). The guide targets 0.1.31-dev.8; its verification scope is documented.

## Development and Testing

```bash
uv run pytest -q
uv run ruff check api tests
npm --prefix web run test:contract
npm --prefix web run build
```

## Important Notes and Disclaimer

- **Protect your secrets**: Never upload `.env`, databases, or Google OAuth client JSON files to a public repository or send them to others. The `ENCRYPTION_KEY` in `.env` decrypts OAuth tokens stored in the database. Keep an offline backup; if you lose it, you will need to authorize your accounts again.
- **Google is the billing authority**: CU usage shown on the website is only an estimate calculated from past jobs. Check your Google Colab account for the actual number of compute units charged.
- **Designed for personal use**: This project is intended for personal use and does not implement multi-tenant isolation. Do not deploy it directly as a public, multi-user SaaS service.
- **License**: This project's code is open source under **GNU AGPL-3.0** (if you modify it and provide a service over a network, you must also make the complete modified source code available). ComfyUI, third-party nodes, model weights, and LoRAs used in workflows remain subject to their respective licenses (no model weight files are included in this repository).

## Support the Community

- [LINUX DO](https://linux.do)

Issues and pull requests are welcome to improve the project or share better workflows! If you find it useful, please consider giving it a Star.

### Reproducible Offline Acceptance Checks (No Real GPU Startup)

After installing dependencies, run `uv run python scripts/verify_offline.py`. This command isolates the configuration, runs backend tests, Ruff, frontend contract checks, and a frontend build. It also starts and stops two temporary Mock services to run browser tests for batches, retries after failure, and long lists. The browser origin must be the local address `127.0.0.1`; use `OFFLINE_CHROMIUM` to specify the Chromium executable. Test logs and network audits are saved in `docs/offline-evidence/`, and `verification-run.json` is written only after all checks pass.

`.github/workflows/offline.yml` provides CI configuration for the same command. Verification has currently been performed locally only, not in GitHub Actions. Dependency installation accesses package repositories, but real Colab, OAuth, Drive operations, and GPU calls are prohibited during acceptance checks. These checks do not establish that real cloud storage or models are usable.
