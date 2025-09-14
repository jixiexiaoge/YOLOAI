# YOLOP 智能驾驶辅助 Android App

## 📱 项目简介

这是一个基于YOLOP模型和ONNX Runtime Mobile的智能驾驶辅助Android应用，能够实时检测道路上的车辆、车道线和可行驶区域。

## 🎯 主要功能

- **实时车辆检测** - 识别道路上的车辆并绘制边界框
- **车道线检测** - 区分实线和虚线车道线
- **可行驶区域识别** - 识别车辆可以行驶的区域
- **实时可视化** - 在摄像头预览上叠加检测结果
- **性能监控** - 显示推理帧率(FPS)

## 🛠️ 技术栈

- **开发语言**: Kotlin
- **UI框架**: Jetpack Compose
- **摄像头**: CameraX API
- **推理引擎**: ONNX Runtime Mobile
- **模型**: YOLOP-320x320 ONNX
- **图像处理**: Android Canvas

## 📋 项目结构

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

## 🚀 快速开始

### 1. 环境要求

- Android Studio Arctic Fox 或更高版本
- Android SDK API 26 (Android 8.0) 或更高
- 支持arm64-v8a架构的设备

### 2. 安装步骤

1. 克隆项目到本地
2. 用Android Studio打开项目
3. 等待Gradle同步完成
4. 连接Android设备或启动模拟器
5. 点击运行按钮

### 3. 权限说明

应用需要以下权限：
- `CAMERA` - 访问摄像头进行实时检测
- `INTERNET` - 网络访问（如需要）
- `WRITE_EXTERNAL_STORAGE` - 存储权限（如需要）

## 📊 性能指标

- **模型**: YOLOP-320x320 ONNX
- **目标设备**: 中端Android手机（骁龙865/870及以上）
- **推理帧率**: ≥ 15 FPS
- **内存占用**: ≤ 500 MB

## 🔧 开发说明

### 当前状态

项目目前使用模拟数据进行演示，包含完整的架构和UI界面。要启用真实的YOLOP模型推理，需要：

1. **集成真正的ONNX Runtime Mobile**
   - 当前使用模拟数据
   - 需要正确配置ONNX Runtime依赖

2. **优化图像处理**
   - 当前使用简化的图像转换
   - 可以集成OpenCV for Android进行更好的图像处理

3. **性能优化**
   - 支持NNAPI硬件加速
   - 多线程处理优化
   - 内存管理优化

### 模块说明

#### CameraManager
- 负责摄像头初始化和图像捕获
- 使用CameraX API确保兼容性
- 提供图像数据流

#### YOLOPModelManager
- 负责ONNX模型加载和推理
- 图像预处理（缩放、归一化）
- 后处理（NMS、坐标转换）

#### ResultRenderer
- 在图像上绘制检测结果
- 支持车辆框、车道线、可行驶区域的可视化
- 显示FPS信息

## 🎨 UI界面

应用采用现代化的Material Design 3设计，采用分屏布局：

### 📱 界面布局
- **上半部分**: 实时摄像头预览画面
  - 显示摄像头实时画面
  - 状态信息覆盖层（运行状态、检测类型）
  
- **下半部分**: AI检测结果展示
  - 车辆检测结果（数量统计）
  - 车道线检测（实线/虚线）
  - 可行驶区域识别状态
  - 实时FPS性能监控

- **底部控制区**: 功能按钮
  - 开始/停止检测
  - 切换显示模式

### 🎯 视觉设计
- **颜色方案**: Material Design 3 动态颜色
- **图标**: 使用Emoji图标增强可读性
- **布局**: 响应式设计，适配不同屏幕尺寸
- **状态指示**: 清晰的视觉反馈和状态提示

## 📝 开发计划

### V1.0 (当前版本)
- [x] 基础架构搭建
- [x] CameraX集成
- [x] UI界面设计
- [x] 模拟数据演示
- [x] 项目编译测试

### V2.0 (计划中)
- [ ] 真实ONNX Runtime集成
- [ ] OpenCV图像处理优化
- [ ] 性能优化和硬件加速
- [ ] 拍照/录像功能
- [ ] 设置界面和参数调整

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

本项目基于MIT许可证开源。

## 🙏 致谢

- [YOLOP](https://github.com/hustvl/YOLOP) - 原始YOLOP模型
- [ONNX Runtime](https://onnxruntime.ai/) - 推理引擎
- [CameraX](https://developer.android.com/training/camerax) - 摄像头API

## 📞 联系方式

如有问题或建议，请提交Issue或联系开发者。

---

**注意**: 当前版本使用模拟数据进行演示。要启用真实的AI推理功能，需要进一步集成ONNX Runtime Mobile和相关优化。
