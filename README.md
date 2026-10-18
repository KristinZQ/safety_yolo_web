# 安全服安全帽检测系统

基于 FastAPI、Ultralytics YOLO 和 OpenCV 的目标检测与人工标注工具。支持 YOLOv11 权重，提供图片检测、视频处理、实时监控和 YOLO 格式标签导出。前端使用原生 HTML、CSS 和 JavaScript。

## 功能

- 单张、批量图片检测，查看和下载检测结果。
- 上传视频后后台逐帧处理，显示任务进度和统计结果。
- 通过本地摄像头、RTSP 或 HTTP 视频流实时检测，使用 WebSocket 传输画面。
- 统计总人数、安全服穿戴情况、未戴安全帽数量、置信度和推理耗时。
- 上传、切换 YOLO 模型，读取模型或 YAML 中的类别，配置报告字段与违规类别。
- 人工框选、撤销、右键删除、保存后自动切换下一张，导出单个标签或 ZIP。
- 根据 PyTorch CUDA 可用性自动选择 GPU 或 CPU。

## 快速开始

环境要求：Python 3.10 及以上，Windows 或 Linux。FFmpeg 为可选依赖，用于生成浏览器兼容的 H.264 视频。

进入项目目录后，在 Windows PowerShell 中执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python run.py
```

Linux 使用：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python run.py
```

访问 [检测页面](http://127.0.0.1:8000) 或 [人工标注页面](http://127.0.0.1:8000/annotation)。接口文档位于 `/docs`。

仓库不包含模型权重。将训练好的 Ultralytics YOLO 权重放到 `weights/best.pt`，或在检测页面底部的模型管理区上传并激活模型。缺少权重时页面和人工标注仍可使用，检测接口会返回模型未加载的提示。

## 配置

配置从项目根目录的 `.env` 读取。常用参数如下：

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `MODEL_PATH` | `weights/best.pt` | 默认模型路径，支持绝对路径 |
| `DEVICE` | 自动选择 | CUDA 可用时为 `cuda:0`，否则为 `cpu` |
| `DEFAULT_CONF` | `0.25` | 置信度阈值 |
| `DEFAULT_IOU` | `0.45` | NMS IoU 阈值 |
| `DEFAULT_IMGSZ` | `640` | 模型输入尺寸 |
| `STREAM_MAX_FPS` | `10` | 实时流处理帧率上限 |
| `MAX_STREAMS` | `2` | 最大实时会话数 |
| `MAX_UPLOAD_MB` | `1024` | 单个上传文件大小上限 |
| `VIOLATION_MODE` | `auto` | 安全服穿戴判定模式 |
| `VEST_OVERLAP_THRESHOLD` | `0.10` | 安全服与人员框的重叠阈值 |
| `VEST_CONF_THRESHOLD` | `0.25` | 安全服判定置信度阈值 |

### GPU

RTX 4060 Ti 等 NVIDIA 显卡在 PyTorch CUDA 可用时自动启用 `cuda:0`。`.env` 中设置 `DEVICE=cpu` 可手动使用 CPU；请求 CUDA 但环境不可用时会回退到 CPU。

检查当前虚拟环境：

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
```

启动和切换模型时，服务日志输出 CUDA 状态、推理设备、GPU 名称与模型路径。当前健康接口提供服务和模型状态，设备信息以启动日志为准。

## 模型与报告映射

检测页面底部的模型管理区支持上传权重、配套 YAML、修改模型类型和切换激活模型。默认模型不能删除，当前激活模型也不能直接删除。切换失败时保留原模型。

模型类别优先读取权重的 `model.names`；提供 YAML 时读取其 `names` 字段。YAML 支持列表或以类别 ID 为键的字典：

```yaml
names:
  0: person
  1: helmet
  2: no_helmet
  3: safety_vest
  4: no_vest
```

类别 ID 和排列顺序必须与训练模型一致。推荐上传 Ultralytics 可加载的 `.pt` 权重；`.pth` 文件不能直接作为检测模型激活。

标签映射用于图片和实时检测下方的信息报告。例如将 `person` 映射为“总人数”，或将自定义类别映射为“植被数量”。检测框继续显示原始模型类别名，映射不提供检测框颜色设置。接口同时保留原始 `class_name` 和报告标签。

`safety_ppe` 模型使用安全穿戴统计，其他类型使用通用目标与分类统计。安全服规则在 `VIOLATION_MODE=auto` 时优先使用模型的 `no_vest` 类；缺少该类但存在人员与安全服类别时，根据人员框内的安全服目标推断穿戴情况。类别别名在 `app/config.py` 的 `CLASS_ALIASES` 中维护。

## 实时监控

视频源填写摄像头编号（如 `0`），或 RTSP、HTTP 流地址：

```text
rtsp://user:password@192.168.1.10:554/stream1
```

摄像头编号对应运行 Python 后端的电脑。使用前关闭占用摄像头的相机、会议或录制程序，并允许桌面应用访问摄像头。运动相机需要切换到 Webcam 模式。

无法取帧时先用系统相机确认设备能输出画面，再关闭系统相机后重试。Windows 下当前使用 OpenCV 自动选择的采集后端；MSMF 取帧失败可能与设备占用、驱动或采集格式有关。RTSP 可先用 VLC 或 FFmpeg 验证地址与码流。

## 人工标注

1. 进入 `/annotation`，上传图片或图片文件夹。
2. 选择或编辑类别；也可从当前模型、YAML 初始化类别。
3. 在图片上按住鼠标左键拖动绘制标注框。
4. 右键点击框边界可删除，撤销按钮可恢复上一步操作。
5. 保存标签后自动切换到下一张未标注图片。
6. 下载当前 `.txt` 或全部 `labels.zip`。

标签坐标按原始图像尺寸归一化，保留 6 位小数：

```text
class_id x_center y_center width height
0 0.512345 0.423456 0.214567 0.158923
```

没有目标的已保存图片生成空标签。导出文件使用原始图片主文件名：

```text
worker_001.jpg -> worker_001.txt
test_image.png -> test_image.txt
```

同一批次出现重名时追加 `_1`、`_2` 等后缀避免覆盖；单文件下载与 ZIP 中使用相同的标签名称。图片内部存储名可以包含 UUID，页面仍显示原始文件名。

## 数据目录

```text
safety_yolo_web/
├── app/
│   ├── main.py              # 应用入口与检测路由
│   ├── config.py            # 环境配置
│   ├── services/            # 图片、视频、实时流和推理
│   ├── routers/             # 标注与模型管理接口
│   ├── annotation/          # 标注存储与 YOLO 标签转换
│   ├── model_registry/      # 模型注册、类别与报告映射
│   ├── utils/               # 文件、绘图与穿戴规则
│   ├── static/              # 检测与标注页面
│   ├── uploads/             # 检测输入
│   ├── outputs/             # 检测结果
│   ├── annotations/         # 类别配置与标注数据集
│   └── model_uploads/       # 上传模型、YAML 与注册表
├── deploy/                 # systemd 与 Nginx 模板
├── weights/                # 默认权重目录
├── .env.example
├── requirements.txt
└── run.py
```

下一次提交图片或视频检测前，前端清理 `app/uploads/` 和 `app/outputs/` 中的历史检测缓存。模型与标注数据独立保存。

虚拟环境、`.env`、模型权重、运行缓存、模型注册表和本地标注配置不纳入版本控制。首次启动会创建运行目录、默认类别与模型注册表。

## 接口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/api/health` | 服务状态、模型加载状态与类别 |
| POST | `/api/detect/image` | 单张图片检测 |
| POST | `/api/detect/images` | 批量图片检测 |
| POST | `/api/detect/video` | 创建视频任务 |
| GET | `/api/tasks/{task_id}` | 视频任务进度与结果 |
| POST | `/api/cache/detection` | 清理历史检测文件 |
| POST | `/api/stream/start` | 创建实时会话 |
| WebSocket | `/ws/stream/{stream_id}` | 接收画面与检测统计 |
| POST | `/api/stream/stop/{stream_id}` | 停止实时会话 |
| GET | `/api/models` | 列出模型 |
| POST | `/api/models/upload` | 上传模型或 YAML |
| POST | `/api/models/{model_id}/mapping` | 保存报告映射 |
| POST | `/api/models/{model_id}/activate` | 激活模型 |
| GET / POST | `/api/annotation/classes` | 读取或保存标注类别 |
| POST | `/api/annotation/upload` | 上传标注图片 |

图片、视频检测接收 multipart 表单，包含 `file`（批量为 `files`）、`conf`、`iou`、`imgsz`。实时会话接收 JSON：

```json
{"source": "0", "conf": 0.25, "iou": 0.45, "imgsz": 640}
```

完整参数和标注数据集接口见运行服务的 `/docs`。

## Linux 部署

将项目放到 `/opt/safety_yolo_web`，创建虚拟环境、安装依赖并配置 `.env` 和权重。`deploy/` 提供 systemd 与 Nginx 模板：

```bash
sudo cp deploy/safety_yolo.service /etc/systemd/system/safety_yolo.service
sudo systemctl daemon-reload
sudo systemctl enable --now safety_yolo
```

按部署环境调整服务用户、目录权限和 Nginx 域名，再启用 `deploy/nginx_safety_yolo.conf`。服务保持单个 worker，避免重复加载模型；Nginx 已包含大文件上传与 WebSocket 代理配置。

当前应用面向本地或可信内网，不包含登录鉴权。对外提供服务时需在反向代理配置访问控制，尤其是模型上传和数据文件接口。

视频输出优先使用 FFmpeg 转为 H.264 MP4；未安装 FFmpeg 时保留 OpenCV `mp4v` 输出，部分浏览器可能无法播放。
