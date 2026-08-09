# YOLO 现成网球模型测试说明

## 当前状态
- 已创建 Python 虚拟环境：`.venv`
- 已安装推理依赖：`ultralytics`
- 已下载 Hugging Face 现成模型：`models/tennis_ball_best.pt`
- 已创建检测模块：`src/tennis_ball_detector.py`
- 已创建图片测试脚本：`scripts/test_yolo_tennis_ball.py`
- 已创建摄像头/视频测试脚本：`scripts/test_yolo_video.py`
- 已完成一次样例图片测试，近处网球可以检出

## 目录说明
```text
远处网球定位/
├── .venv/
├── models/
│   └── tennis_ball_best.pt
├── inputs/
│   └── sample_single_ball.jpg
├── outputs/
│   ├── detections.csv
│   └── sample_single_ball_detected.jpg
├── src/tennis_ball_detector.py
├── scripts/test_yolo_video.py
├── scripts/test_yolo_tennis_ball.py
└── YOLO现成模型测试说明.md
```

## 检测模块作用
`src/tennis_ball_detector.py` 是后面复用的核心模块。

输入：

```text
OpenCV 图像帧
```

输出：

```text
bbox
center_x
center_y
confidence
```

后续接 ROS2、深度图、3D 坐标时，都应调用这个模块，而不是直接把 YOLO 代码写在业务节点里。

## 图片测试
在当前目录执行：

```powershell
.\.venv\Scripts\python.exe scripts/test_yolo_tennis_ball.py --source inputs --out outputs --conf 0.25
```

## 输出结果
- 带检测框的图片会保存到 `outputs/`
- 检测框数据会保存到 `outputs/detections.csv`

`detections.csv` 包含：
- `confidence`：置信度
- `x1, y1, x2, y2`：检测框左上角和右下角像素坐标
- `center_x, center_y`：检测框中心点像素坐标

## 摄像头/视频测试
默认使用电脑第 0 个摄像头：

```powershell
.\.venv\Scripts\python.exe scripts/test_yolo_video.py --source 0 --conf 0.25
```

如果外接摄像头，可以改成：

```powershell
.\.venv\Scripts\python.exe scripts/test_yolo_video.py --source 1 --conf 0.25
```

或：

```powershell
.\.venv\Scripts\python.exe scripts/test_yolo_video.py --source 2 --conf 0.25
```

也可以直接测试视频文件：

```powershell
.\.venv\Scripts\python.exe scripts/test_yolo_video.py --source test_video.mp4 --conf 0.25
```

如果想修改默认摄像头，改 `scripts/test_yolo_video.py` 顶部：

```python
DEFAULT_CAMERA_SOURCE = "0"
```

例如外接摄像头固定是 1，就改成：

```python
DEFAULT_CAMERA_SOURCE = "1"
```

运行时按 `q` 退出。

## 树莓派部署时的图片保存策略
摄像头测试脚本默认不保存每一帧图片，只显示检测结果。

原因：
- 避免占满树莓派 SD 卡
- 减少磁盘写入
- 减少实时检测延迟

只有需要调试时才加：

```powershell
.\.venv\Scripts\python.exe scripts/test_yolo_video.py --source 0 --save-debug
```

调试图片默认保存到：

```text
outputs/video_debug/
```

## 测试你自己的图片
把真实网球场图片放进：

```text
inputs/
```

然后重新运行：

```powershell
.\.venv\Scripts\python.exe scripts/test_yolo_tennis_ball.py --source inputs --out outputs --conf 0.25
```

## 重点测试图片
- 近处球
- 远处球
- 多个球
- 球靠近白线
- 球在阴影里
- 强光照射
- 球只露出一部分

## 判断标准
- 远处球能不能检出
- 白线会不会被误检成球
- 黄色物体会不会被误检成球
- 多个球时是否能全部框出
- 检测框中心点是否落在球附近

