# 环境与启动

## 本机开发

使用已创建的 `sop-detector` Conda 环境，不再套用 venv。先安装 requirements-gpu.txt，再安装 requirements.txt。当前锁定 PyTorch 2.7.1+cu126 / torchvision 0.22.1+cu126，避免默认安装包替换已验证的 MX150 环境；无需为这些预编译包另装 CUDA Toolkit。

```powershell
conda activate sop-detector
cd D:\projects\SOP_Detecter
python -m pip install -r requirements-gpu.txt *> D:\projects\logs\sop-detector-gpu-install.log
python -m pip install -r requirements.txt *> D:\projects\logs\sop-detector-install.log
python scripts/download_model.py *> D:\projects\logs\sop-detector-model-download.log
python start.py
```

在 VS Code 中选择 Conda `sop-detector` 解释器，打开工程根目录的 `start.py`，点击“运行 Python 文件”即可。启动文件不绑定某个用户的 Python 安装路径，也不需要命令行参数。配置文件路径由 `start.py` 的 `CONFIG_FILE` 指定，端口、模型、设备和日志目录仍在 JSON 中维护。默认端口 18092，独立于 Java 8080 和 Go 18090。Ctrl+C 停止后 Uvicorn 等待请求结束并释放模型。

`start.py` 使用相对自身的绝对配置路径，从其他工作目录运行也可以。运行日志写统一日志目录。需要归档终端输出时，也重定向至同一目录。不要同时使用多个 worker 或开发热重载，否则会重复占用显存。原 PowerShell 启动脚本已移除。

配置中的相对权重路径相对工程根目录解析，和启动时的工作目录无关。权重缺失不会隐式下载。下载脚本只取官方 v8.3.0 发布的 yolov8m.pt，并校验固定 SHA256：
`f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`。

## 检查顺序

1. `GET /health` 检查存活及 models_ready。
2. 用凭证请求 `/models`，核对模型状态、版本和真实类别。
3. `/docs` 中 Authorize 填写开发凭证，上传样图，metadata 填 `{"model_id":"yolov8m","frame_id":"test-1"}`。
4. 先验证单张图片，再由浏览器以低频发送视频帧。网络传输与前端绘制不计入 inference_ms。

模型 load_failed 时检查权重类型、GPU 可用性和显存。当前版本不自动从 CUDA 切到 CPU，需明确配置 device=cpu 并重启。2GB 显存先保持一个 nano 模型；增加模型前实测常驻显存和延迟。

## 依赖与授权

FastAPI / Uvicorn 提供 HTTP，Ultralytics YOLO 执行目标检测，模型按官方 Python predict API 调用：
https://docs.ultralytics.com/modes/predict/
服务生命周期使用 FastAPI lifespan：
https://fastapi.tiangolo.com/advanced/events/

Ultralytics 及其模型采用对应的 AGPL-3.0 / 企业许可条款，分发和商业部署前按实际使用方式确认许可；当前开发工程不代替上游授权。
