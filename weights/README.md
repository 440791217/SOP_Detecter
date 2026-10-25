# 模型权重

这里存放可信的 `.pt` 目标检测模型，权重不提交 Git。

当前配置使用 `yolov8m.pt`，官方来源：
https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt

这是 COCO 80 类的通用模型，可用于开发验证，不能直接当作螺丝、接插件或装配状态的专用模型。
新增模型时，在 `configs/settings.json` 的 `models` 中登记唯一 ID、版本和文件路径，重启服务后加载。
只加载可信来源的 PyTorch 权重；当前不接受网页上传模型、请求指定本地模型路径或运行时自动下载模型。

当前仅启用 YOLOv8m，输入尺寸 640，单进程单次推理。旧 yolov8n.pt 文件保留用于回退，不同时加载。
