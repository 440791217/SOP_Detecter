"""直接运行本文件启动检测服务，无需命令行参数。"""
from pathlib import Path

from app.main import main

# 配置文件路径在这里指定；相对本文件定位，不依赖终端当前目录。
CONFIG_FILE = Path(__file__).resolve().parent / "configs" / "settings.json"


if __name__ == "__main__":
    main(CONFIG_FILE)
