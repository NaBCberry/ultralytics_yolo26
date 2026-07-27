# 钢球识别网页服务

`steelball_web.py` 是独立的网页服务，不会修改现有的命令行推理脚本。它默认使用仓库中的钢球实例分割模型，并将检测框和中心点叠加到视频流与拍摄照片上。

## 启动

在开发板上运行：

```bash
cd runtime/python
chmod +x run_steelball_web.sh
./run_steelball_web.sh
```

然后从同一网络的电脑或手机访问：

```text
http://<开发板IP>:5000
```

默认使用摄像头 `0`、分辨率 `1280x720` 与模型 `../../model/steelball_seg_bpu_bayese_640x640_nv12.bin`。

## 常用参数

```bash
./run_steelball_web.sh --camera /dev/video0 --width 1920 --height 1080 --camera-format MJPG --camera-fps 30
./run_steelball_web.sh --stream-width 640 --stream-jpeg-quality 75
./run_steelball_web.sh --port 8080 --score-thres 0.35
./run_steelball_web.sh --model-path /path/to/model.bin --model-type detect
```

- `--camera`：摄像头索引或设备文件路径。
- `--camera-format`：默认请求 `MJPG`，减少 USB 带宽占用；仅在摄像头不支持 MJPG 时使用 `YUYV`。
- `--camera-fps`：默认请求 `30` FPS。服务启动日志会输出摄像头最终协商到的分辨率、帧率和像素格式。
- `--mjpeg-passthrough`：默认启用。拍摄页面将摄像头的原始 MJPG 直接推送给浏览器，避免 OpenCV 的逐帧解码和重新编码；不支持时服务会自动回退并输出警告日志。
- `--stream-width`：拍摄页面的网页预览宽度，默认 `640`，用于降低 JPEG 编码负载；拍摄照片和模型输入仍为摄像头原始分辨率。
- `--stream-jpeg-quality`：网页预览 JPEG 质量，默认 `75`；在帧率仍不足时可再降低，例如设为 `60`。
- `--model-type`：默认 `seg`；当使用钢球检测模型时设为 `detect`。
- `--bpu-cores`：默认使用 BPU Core 0，例如 `--bpu-cores 0 1`。

网页包含两个可切换的功能：

- 实时识别模式持续运行模型，显示带识别框的视频、数量与实时中心点坐标。
- 拍摄识别模式只显示干净的实时视频流，后台不会执行连续推理。点击拍摄后，右侧先显示原始照片，标题显示“已获取图像，识别中”；完成单次识别后，照片更新为带识别框的结果，标题显示绿色“识别成功”，并显示本次钢球数量和中心点坐标。
