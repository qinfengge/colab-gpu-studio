"""Initialize private configuration once. Existing files are never overwritten."""

import argparse
import os
import secrets
from pathlib import Path

from cryptography.fernet import Fernet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=".env")
    p.add_argument("--site-url", default="http://localhost:8788")
    a = p.parse_args()
    path = Path(a.output)
    if path.exists():
        raise SystemExit("配置已存在；为保护凭据不会覆盖。")
    values = {
        "ENVIRONMENT": "development",
        "SITE_URL": a.site_url.rstrip("/"),
        "DATABASE_URL": "sqlite:///./data/studio.db",
        "DATA_DIR": "./data",
        "STUDIO_PASSWORD": secrets.token_urlsafe(24),
        "ENCRYPTION_KEY": Fernet.generate_key().decode(),
        "SESSION_SECRET": secrets.token_urlsafe(48),
        "STUDIO_API_TOKEN": secrets.token_urlsafe(40),
        "GPU_AGENT_TOKEN": secrets.token_urlsafe(40),
        "GPU_AGENT_OWNER_ID": "owner",
        "GPU_REAL_ENABLED": "0",
        "GPU_STORAGE": "drive",
        "GPU_COMPUTE_PROVIDER": "kaggle",
        "GPU_MODEL_CACHE": "kaggle",
        "KAGGLE_USERNAME": "",
        "KAGGLE_API_TOKEN": "",
        "KAGGLE_MODEL_DATASETS": "",
        "KAGGLE_DRIVE_ACCOUNT_ID": "",
        "GPU_MAX_RUNTIME_MINUTES": "30",
        "GPU_DAILY_UNITS": "10",
        "GPU_MONTHLY_UNITS": "100",
        "GPU_MAX_JOB_UNITS": "5",
        "GPU_UNIT_RATE": "",
        "GPU_GOOGLE_CLIENT_ID": "",
        "GPU_GOOGLE_CLIENT_SECRET": "",
        "GPU_GOOGLE_REDIRECT_URI": "",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(k + "=" + v for k, v in values.items()) + "\n")
    print("已创建私密配置：" + str(path) + "；密码在文件内，不在日志回显。真实GPU默认关闭。")


if __name__ == "__main__":
    main()
