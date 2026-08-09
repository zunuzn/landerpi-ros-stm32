# 当前 GPU 环境总结

生成时间：2026-08-06

## 项目位置

```text
C:\Users\ASUS\Desktop\slam\远处网球定位
```

## 当前 YOLO 模型

- 使用框架：Ultralytics YOLO
- 模型文件：`models/tennis_ball_best.pt`
- 任务类型：目标检测 detect
- 检测类别：`tennis-ball`
- 模型大小：约 6.3 MB

## 显卡和驱动

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- 显存：8188 MiB
- NVIDIA 驱动版本：566.07
- `nvidia-smi` 显示 CUDA 版本：12.7

## Python 虚拟环境

- 虚拟环境路径：`.venv`
- Python 版本：3.14.6
- 运行解释器：

```powershell
.\.venv\Scripts\python.exe
```

## 深度学习环境

当前已从 CPU 版 PyTorch 切换到 CUDA 版 PyTorch：

- PyTorch：`2.13.0+cu126`
- Torch CUDA runtime：`12.6`
- TorchVision：`0.28.0+cu126`
- TorchAudio：`2.11.0+cu126`
- Ultralytics：`8.4.115`

验证结果：

```text
torch.cuda.is_available() = True
device_count = 1
device0 = NVIDIA GeForce RTX 4060 Laptop GPU
```

## 已修改的代码

以下脚本已支持 `--device` 参数：

- `src/tennis_ball_detector.py`
- `scripts/test_yolo_video.py`
- `scripts/test_yolo_tennis_ball.py`

`--device auto` 会自动选择：

- 有 CUDA 时使用 `cuda:0`
- 没有 CUDA 时使用 `cpu`

## 推荐运行命令

视频检测：

```powershell
.\.venv\Scripts\python.exe scripts/test_yolo_video.py --source court_test.mp4 --device cuda:0
```

或者使用自动选择：

```powershell
.\.venv\Scripts\python.exe scripts/test_yolo_video.py --source court_test.mp4 --device auto
```

摄像头检测：

```powershell
.\.venv\Scripts\python.exe scripts/test_yolo_video.py --source 0 --device cuda:0
```

图片检测：

```powershell
.\.venv\Scripts\python.exe scripts/test_yolo_tennis_ball.py --source inputs --out outputs --device cuda:0
```

## 验证命令

检查 PyTorch 是否能使用 GPU：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

检查 YOLO 检测器自动选择的设备：

```powershell
.\.venv\Scripts\python.exe -c "from src.tennis_ball_detector import TennisBallDetector; d=TennisBallDetector(model_path='models/tennis_ball_best.pt', device='auto'); print(d.device)"
```

当前验证结果：

```text
device cuda:0
detections 1
ok True
```

说明 YOLO 已经可以在 GPU 上完成一次视频帧推理。

## 注意事项

- GPU 加速已经生效，但 1080p、60 FPS 视频仍然可能很吃性能。
- 如果还觉得慢，下一步应优先做缩放推理、跳帧检测、降低显示分辨率。
- 树莓派上不能直接使用这套 CUDA 配置，树莓派部署需要考虑 ONNX、NCNN、OpenVINO 或其他轻量推理方案。

