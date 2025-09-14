# YOLOAI 项目说明

## 项目概述

这是一个基于 YOLOP 模型和 ONNX Runtime Mobile 的智能驾驶辅助 Android 应用。该应用能够实时检测道路上的车辆、车道线（区分实线和虚线）和可行驶区域，并在摄像头预览上叠加检测结果。

主要技术栈：
- **开发语言**: Kotlin
- **UI框架**: Jetpack Compose
- **摄像头**: CameraX API
- **推理引擎**: ONNX Runtime Mobile
- **模型**: YOLOP-320x320 ONNX

## 项目结构

```
app/src/main/java/com/example/yoloai/
├── MainActivity.kt                    # 主活动，集成所有模块
├── camera/
│   └── CameraManager.kt              # CameraX管理器
├── onnx/
│   └── YOLOPModelManager.kt          # ONNX模型管理器
└── visualization/
    └── ResultRenderer.kt             # 结果渲染器
```

## 构建和运行

### 环境要求
- Android Studio Arctic Fox 或更高版本
- Android SDK API 26 (Android 8.0) 或更高
- 支持 arm64-v8a 架构的设备

### 安装步骤
1. 克隆项目到本地
2. 用 Android Studio 打开项目
3. 等待 Gradle 同步完成
4. 连接 Android 设备或启动模拟器
5. 点击运行按钮

### 构建命令
在项目根目录下执行：
```bash
# 构建 debug 版本
./gradlew assembleDebug

# 构建 release 版本
./gradlew assembleRelease

# 运行单元测试
./gradlew test

# 运行 Android instrumentation 测试
./gradlew connectedAndroidTest
```

## 开发约定

### 代码风格
- 使用 Kotlin 作为主要开发语言
- 遵循 Android 官方 Kotlin 代码风格指南
- 使用 Jetpack Compose 构建 UI
- 使用协程进行异步操作

### 模块说明

#### CameraManager
负责摄像头初始化和图像捕获，使用 CameraX API 确保兼容性。

#### YOLOPModelManager
负责 ONNX 模型加载和推理，包括图像预处理、模型推理和结果后处理。

#### ResultRenderer
在图像上绘制检测结果，支持车辆框、车道线、可行驶区域的可视化，并显示 FPS 信息。

### 当前状态
项目目前使用真实的 YOLOP ONNX 模型进行推理，包含完整的架构和 UI 界面。后续可以进一步优化性能和增加新功能。