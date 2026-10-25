# SOP_Detecter

轻量 Python 目标检测服务：接收浏览器图片，复用常驻模型，返回检测框。单进程、单次推理，无数据库和消息队列。

## 已验证环境

Windows 10 x64、Conda `sop-detector`、Python 3.13、PyTorch 2.7.1+cu126、torchvision 0.22.1+cu126、MX150 2GB。当前只支持 Ultralytics `.pt` 目标检测模型。

## 快速启动

```powershell
cd D:\projects\SOP_Detecter
conda activate sop-detector
python -m pip install -r requirements-gpu.txt
python -m pip install -r requirements.txt
python scripts/download_model.py
python start.py
```

当前机器已经安装依赖、下载并验证开发权重，可以直接在 VS Code 中打开根目录的 `start.py`，点击“运行 Python 文件”。配置文件路径已在 `start.py` 的 `CONFIG_FILE` 中指定，不需要额外传参。

- 接口文档：http://localhost:18092/docs
- 健康状态：http://localhost:18092/health
- 开发 Bearer 凭证：`88888888`，可在 `configs/settings.json` 中修改。
- 运行和启动输出日志全部位于 `D:\projects\logs`。
- 默认监听本机；开发网页允许来源为 localhost:9527、127.0.0.1:9527。

`/docs` 的 Authorize 中填写凭证后，可以上传 JPEG/PNG 测试预测。
API 不拉 RTSP，不保存输入照片，不生成标注图片，也不负责推进 SOP 步骤。
YOLOv8m 是通用 COCO 模型，不是工业工件专用模型。

## 文档

- [总体设计](docs/design.md)
- [接口约定](docs/api.md)
- [安装、部署与排错](docs/deployment.md)
- [权重放置说明](weights/README.md)

## 测试

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests/test_service.py -q -p no:cacheprovider *> D:\projects\logs\sop-detector-test.log
$env:SOP_DETECTOR_GPU_TEST='1'
python -m pytest tests/test_gpu.py -s -q -p no:cacheprovider *> D:\projects\logs\sop-detector-gpu-test.log
```

接口测试使用明确的替身模型，GPU 测试才使用当前启用的真实模型和官方 bus.jpg 样图。替身模型不会进入运行服务。
