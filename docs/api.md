# HTTP 接口

开发地址：`http://localhost:18092`。Swagger UI：`/docs`。
`/models` 和 `/predict` 需要 `Authorization: Bearer 88888888`。配置文件可更换凭证。

## GET /health

无需凭证。HTTP 200 表示 HTTP 服务存活，不代表模型可用。`models_ready` 表示至少一个启用模型且全部启用模型 ready。

```json
{"service":"sop-detector","status":"ok","models_ready":true}
```

## GET /models

返回执行设备、模型 ID、版本、状态和模型真实类别列表。状态包括 not_loaded、disabled、ready、unavailable、stopped；error 可能为 weights_missing 或 load_failed。不会返回权重文件的本地路径。

## POST /predict

Content-Type 为浏览器生成的 multipart/form-data。不要手工固定 boundary。

| 表单字段 | 类型 | 说明 |
|---|---|---|
| file | 文件 | JPEG/PNG，默认不超过 5MiB |
| metadata | JSON 字符串 | 下述请求上下文，最大 8192 字符 |

metadata 示例：

```json
{"model_id":"yolov8m","frame_id":"frame-100","station_id":"7","execution_id":"9","step_id":"3","confidence":0.25}
```

model_id 和 frame_id 必填；其他上下文可选，业务联调时应完整填写。confidence 在 0～1 范围内，未填使用模型配置。业务中的百分比置信度应先除以 100。未知字段、空必填 ID 和非法值均拒绝。

成功 HTTP 200 示例：

```json
{
  "context":{"model_id":"yolov8m","frame_id":"frame-100","station_id":"7","execution_id":"9","step_id":"3","confidence":0.25},
  "model_version":"coco-v8.3.0",
  "image_width":1280,
  "image_height":720,
  "inference_ms":42.5,
  "total_ms":47.8,
  "detections":[{"class_id":0,"class_name":"person","confidence":0.92,"bbox_xyxy":[100,80,300,500]}]
}
```

坐标为校正 EXIF 后原图的左上、右下像素坐标；confidence 为 0～1；耗时为毫秒。类别数字只对该模型及版本有效，不跨模型当作同一个业务类别。没有目标时 detections 为空数组，不代表步骤成功。

## 状态码

| 状态 | detail | 前端处理 |
|---|---|---|
| 400 | invalid_image / empty_image | 换新帧 |
| 401 | invalid_access_token | 检查检测服务凭证 |
| 403 | origin_not_allowed | 检查允许的网站来源 |
| 404 | model_not_found | 检查模型 ID |
| 413 | image_too_large / image_dimensions_too_large / request_too_large | 降低上传图片尺寸 |
| 415 | only_jpeg_and_png_supported | 使用 JPEG/PNG |
| 422 | invalid_prediction_metadata 或框架字段错误 | 修正表单参数 |
| 429 | detector_busy | 丢弃旧帧，稍后取最新帧 |
| 503 | model_not_ready / inference_failed | 停止自动推进并显示未就绪 |

服务不提供训练、上传模型、拉流、步骤确认或图片持久化接口。
