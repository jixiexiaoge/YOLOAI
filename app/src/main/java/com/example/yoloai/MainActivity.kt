package com.example.yoloai

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.yoloai.camera.CameraManager
import com.example.yoloai.onnx.YOLOPModelManager
import com.example.yoloai.onnx.YOLOPResult
import com.example.yoloai.ui.theme.YOLOAITheme
import com.example.yoloai.visualization.ResultRenderer
import kotlinx.coroutines.launch
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.consumeEach
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean

/**
 * 主活动
 * 集成摄像头、模型推理和结果可视化
 */
class MainActivity : ComponentActivity() {
    
    companion object {
        private const val TAG = "MainActivity"
    }
    
    private lateinit var cameraManager: CameraManager
    private lateinit var modelManager: YOLOPModelManager
    private lateinit var resultRenderer: ResultRenderer
    
    private var isModelInitialized = false
    private var isProcessing = false
    
    // 异步处理架构 - 优化版本
    private val imageProcessingChannel = Channel<Bitmap>(capacity = 1) // 进一步减少队列大小，避免积压
    private val isProcessingActive = AtomicBoolean(false)
    private var processingJob: kotlinx.coroutines.Job? = null
    
    // 帧跳过计数器
    private var skippedFrames = 0
    private var processedFrames = 0
    
    // 性能监控
    private var lastFrameTime = System.currentTimeMillis()
    private var processingFrameCount = 0
    private var actualFps = 0f
    private var lastFpsTime = System.currentTimeMillis()
    private var currentFps = 0f
    
    // 数据源类型
    enum class DataSource {
        CAMERA,  // 手机摄像头
        STREAM   // 视频流
    }
    
    // UI状态管理
    private var previewView: PreviewView? = null
    private var processedBitmap: Bitmap? = null
    
    // 用于触发UI更新的状态
    private var uiUpdateTrigger = 0
    
    // 使用MutableState来触发Compose重新组合
    private val _processedBitmapState = mutableStateOf<Bitmap?>(null)
    private val _uiUpdateTriggerState = mutableStateOf(0)
    private val _dataSourceState = mutableStateOf(DataSource.CAMERA)
    private val _ipAddressState = mutableStateOf("")
    private val _showIpInputState = mutableStateOf(false)
    private val _currentResolutionState = mutableStateOf(YOLOPModelManager.Companion.InputResolution.RESOLUTION_320)
    private val _showResolutionSelector = mutableStateOf(false)
    
    // 权限请求
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            Log.i(TAG, "摄像头权限已授予")
            initializeCamera()
        } else {
            Log.e(TAG, "摄像头权限被拒绝")
            Toast.makeText(this, "需要摄像头权限才能使用此应用", Toast.LENGTH_LONG).show()
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        
        // 设置屏幕常亮
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        
        // 初始化组件
        cameraManager = CameraManager(this, this)
        modelManager = YOLOPModelManager(this)
        resultRenderer = ResultRenderer()
        
        setContent {
            YOLOAITheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    YOLOPApp(
                        previewView = previewView,
                        processedBitmap = _processedBitmapState.value,
                        uiUpdateTrigger = _uiUpdateTriggerState.value,
                        dataSource = _dataSourceState.value,
                        ipAddress = _ipAddressState.value,
                        showIpInput = _showIpInputState.value,
                        onDataSourceChange = { newSource ->
                            _dataSourceState.value = newSource
                            if (newSource == DataSource.STREAM) {
                                _showIpInputState.value = true
                            } else {
                                _showIpInputState.value = false
                                // 切换到摄像头模式时重新初始化摄像头
                                initializeCamera()
                            }
                        },
                        onIpAddressChange = { newIp ->
                            _ipAddressState.value = newIp
                        },
                        onConnectStream = { ip ->
                            connectStream(ip)
                        },
                        onCancelIpInput = {
                            _showIpInputState.value = false
                            _dataSourceState.value = DataSource.CAMERA
                        },
                        currentResolution = _currentResolutionState.value,
                        showResolutionSelector = _showResolutionSelector.value,
                        onResolutionChange = { newResolution ->
                            _currentResolutionState.value = newResolution
                            modelManager.setInputResolution(newResolution)
                            _showResolutionSelector.value = false
                        },
                        onToggleResolutionSelector = {
                            _showResolutionSelector.value = !_showResolutionSelector.value
                        }
                    )
                }
            }
        }
        
        // 检查权限并初始化
        checkCameraPermission()
    }
    
    /**
     * 检查摄像头权限
     */
    private fun checkCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                Log.i(TAG, "摄像头权限已授予")
                initializeCamera()
            }
            else -> {
                Log.i(TAG, "请求摄像头权限")
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }
    
    /**
     * 初始化摄像头
     */
    private fun initializeCamera() {
        lifecycleScope.launch {
            // 创建一个PreviewView实例
            val previewView = PreviewView(this@MainActivity).apply {
                scaleType = PreviewView.ScaleType.FILL_CENTER
                layoutParams = ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                )
                // 设置旋转以匹配设备方向
                rotation = 0f
            }
            
            val success = cameraManager.initialize(previewView)
            if (success) {
                Log.i(TAG, "摄像头初始化成功")
                // 将PreviewView设置到UI中
                setPreviewView(previewView)
                initializeModel()
            } else {
                Log.e(TAG, "摄像头初始化失败")
                Toast.makeText(this@MainActivity, "摄像头初始化失败", Toast.LENGTH_LONG).show()
            }
        }
    }
    
    /**
     * 初始化模型
     */
    private fun initializeModel() {
        lifecycleScope.launch {
            val success = modelManager.initialize()
            if (success) {
                Log.i(TAG, "模型初始化成功")
                isModelInitialized = true
                startProcessing()
            } else {
                Log.e(TAG, "模型初始化失败")
                Toast.makeText(this@MainActivity, "模型加载失败", Toast.LENGTH_LONG).show()
            }
        }
    }
    
    /**
     * 开始异步图像处理
     */
    private fun startProcessing() {
        // 启动异步处理协程
        processingJob = lifecycleScope.launch(Dispatchers.Default) {
            startAsyncImageProcessing()
        }
        
        // 启动图像采集协程 - 智能优化版本
        lifecycleScope.launch(Dispatchers.Main) {
            var frameCount = 0
            cameraManager.imageFlow.collect { bitmap ->
                frameCount++
                if (isModelInitialized) {
                    // 智能帧跳过：如果最近跳过的帧太多，暂时停止采集
                    if (skippedFrames > processedFrames * 5 && skippedFrames > 20) {
                        Log.d(TAG, "跳帧过多，暂停采集")
                        return@collect
                    }
                    
                    // 非阻塞地将图像放入处理队列
                    if (!imageProcessingChannel.isClosedForSend) {
                        val result = imageProcessingChannel.trySend(bitmap)
                        if (result.isFailure) {
                            // 队列满了，跳过这一帧
                            skippedFrames++
                            if (skippedFrames % 10 == 0) {
                                Log.d(TAG, "已跳过 $skippedFrames 帧，处理了 $processedFrames 帧")
                            }
                        } else {
                            processedFrames++
                        }
                    }
                }
            }
        }
    }
    
    /**
     * 异步图像处理协程 - 优化版本
     */
    private suspend fun startAsyncImageProcessing() {
        imageProcessingChannel.consumeEach { bitmap ->
            if (isModelInitialized && isProcessingActive.compareAndSet(false, true)) {
                try {
                    val processStart = System.currentTimeMillis()
                    
                    withContext(Dispatchers.IO) {
                        processImage(bitmap)
                    }
                    
                    val processTime = System.currentTimeMillis() - processStart
                    
                    // 计算实际FPS
                    processingFrameCount++
                    val currentTime = System.currentTimeMillis()
                    if (currentTime - lastFrameTime >= 1000) { // 每秒计算一次
                        actualFps = processingFrameCount * 1000.0f / (currentTime - lastFrameTime)
                        Log.i(TAG, "实际处理FPS: ${String.format("%.1f", actualFps)}, 处理时间: ${processTime}ms")
                        processingFrameCount = 0
                        lastFrameTime = currentTime
                    }
                    
                } catch (e: Exception) {
                    Log.e(TAG, "异步处理图像时出错: ${e.message}", e)
                } finally {
                    isProcessingActive.set(false)
                }
            }
        }
    }
    
    /**
     * 处理图像
     */
    private suspend fun processImage(bitmap: Bitmap) {
        try {
            val result = modelManager.inference(bitmap)
            if (result != null) {
                val renderedBitmap = resultRenderer.renderResults(bitmap, result)
                
                // 强制在主线程更新UI状态，确保UI更新
                runOnUiThread {
                    // 关键：回收旧的Bitmap以释放内存，避免内存泄漏和卡顿
                    _processedBitmapState.value?.recycle()
                    _processedBitmapState.value = renderedBitmap
                    // 只在结果有显著变化时才更新UI触发次数
                    _uiUpdateTriggerState.value = _uiUpdateTriggerState.value + 1
                    Log.d(TAG, "UI状态已更新 - 触发次数: ${_uiUpdateTriggerState.value}, FPS: ${result.fps}")
                }
                
                // 计算实际FPS
                processingFrameCount++
                val currentTime = System.currentTimeMillis()
                if (currentTime - lastFpsTime >= 1000) { // 每秒计算一次FPS
                    currentFps = processingFrameCount * 1000.0f / (currentTime - lastFpsTime)
                    processingFrameCount = 0
                    lastFpsTime = currentTime
                    Log.i(TAG, "实际FPS: ${String.format("%.1f", currentFps)}, 模型FPS: ${String.format("%.1f", result.fps)}, 检测数量: ${result.detections.size}")
                }
                
                Log.d(TAG, "处理完成 - 模型FPS: ${result.fps}, 检测数量: ${result.detections.size}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "图像处理失败: ${e.message}", e)
        }
    }
    
    /**
     * 设置预览视图
     */
    private fun setPreviewView(previewView: PreviewView) {
        this.previewView = previewView
    }
    
    /**
     * 连接视频流
     */
    private fun connectStream(ipAddress: String) {
        lifecycleScope.launch {
            try {
                Log.i(TAG, "开始连接视频流: $ipAddress")
                
                // 这里可以添加实际的视频流连接逻辑
                // 目前只是模拟连接
                Toast.makeText(this@MainActivity, "连接视频流: $ipAddress", Toast.LENGTH_SHORT).show()
                
                _showIpInputState.value = false
                Log.i(TAG, "视频流连接已启动")
                
            } catch (e: Exception) {
                Log.e(TAG, "视频流连接失败: ${e.message}", e)
                Toast.makeText(this@MainActivity, "视频流连接失败: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }
    
    
    override fun onDestroy() {
        super.onDestroy()
        
        // 停止异步处理
        processingJob?.cancel()
        imageProcessingChannel.close()
        
        // 释放资源
        cameraManager.release()
        modelManager.release()
    }
}

/**
 * 主应用UI
 */
@Composable
fun YOLOPApp(
    previewView: PreviewView?,
    processedBitmap: Bitmap?,
    uiUpdateTrigger: Int,
    dataSource: MainActivity.DataSource,
    ipAddress: String,
    showIpInput: Boolean,
    onDataSourceChange: (MainActivity.DataSource) -> Unit,
    onIpAddressChange: (String) -> Unit,
    onConnectStream: (String) -> Unit,
    onCancelIpInput: () -> Unit,
    currentResolution: YOLOPModelManager.Companion.InputResolution,
    showResolutionSelector: Boolean,
    onResolutionChange: (YOLOPModelManager.Companion.InputResolution) -> Unit,
    onToggleResolutionSelector: () -> Unit
) {
    val lifecycleOwner = LocalLifecycleOwner.current
    
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // 标题
        Text(
            text = "YOLOP 智能驾驶辅助",
            style = MaterialTheme.typography.headlineMedium,
            modifier = Modifier.padding(16.dp)
        )
        
        
        // 主摄像头预览区域（320x320固定尺寸）
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp),
            contentAlignment = Alignment.Center
        ) {
            // 摄像头预览 - 固定320x320尺寸
            if (previewView != null) {
                AndroidView(
                    factory = { previewView!! },
                    modifier = Modifier.size(320.dp)
                )
            } else {
                // 加载中显示
                Box(
                    modifier = Modifier.size(320.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        CircularProgressIndicator()
                        Spacer(modifier = Modifier.height(16.dp))
                        Text("正在初始化摄像头...")
                    }
                }
            }
            
            // 状态信息覆盖层
            Card(
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.9f)
                )
            ) {
                Column(
                    modifier = Modifier.padding(12.dp)
                ) {
                    Text(
                        text = "📹 实时摄像头",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.primary
                    )
                    Text(
                        text = "状态: 运行中",
                        style = MaterialTheme.typography.bodySmall
                    )
    Text(
                        text = "检测: 车辆、车道线、可行驶区域",
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }
        }
        
        // IP地址输入对话框
        if (showIpInput) {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 8.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            ) {
                Column(
                    modifier = Modifier.padding(16.dp)
                ) {
                    Text(
                        text = "输入视频流IP地址",
                        style = MaterialTheme.typography.titleMedium,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )
                    
                    OutlinedTextField(
                        value = ipAddress,
                        onValueChange = onIpAddressChange,
                        label = { Text("IP地址 (例如: 192.168.1.100)") },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(bottom = 8.dp)
                    )
                    
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.End
                    ) {
                        TextButton(onClick = onCancelIpInput) {
                            Text("取消")
                        }
                        Spacer(modifier = Modifier.width(8.dp))
                        Button(
                            onClick = { onConnectStream(ipAddress) },
                            enabled = ipAddress.isNotBlank()
                        ) {
                            Text("连接")
                        }
                    }
                }
            }
        }
        
        // 切换按钮区域
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                // 数据源切换按钮
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceEvenly,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    // 摄像头模式按钮
                    Button(
                        onClick = { onDataSourceChange(MainActivity.DataSource.CAMERA) },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (dataSource == MainActivity.DataSource.CAMERA) 
                                MaterialTheme.colorScheme.primary 
                            else MaterialTheme.colorScheme.surface
                        )
                    ) {
                        Text("📹 摄像头")
                    }
                    
                    Spacer(modifier = Modifier.width(8.dp))
                    
                    // 视频流模式按钮
                    Button(
                        onClick = { onDataSourceChange(MainActivity.DataSource.STREAM) },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (dataSource == MainActivity.DataSource.STREAM) 
                                MaterialTheme.colorScheme.primary 
                            else MaterialTheme.colorScheme.surface
                        )
                    ) {
                        Text("🌐 视频流")
                    }
                }
                
                Spacer(modifier = Modifier.height(8.dp))
                
                // 分辨率选择按钮
                Button(
                    onClick = onToggleResolutionSelector,
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.secondary
                    )
                ) {
                    Text("🎯 分辨率: ${currentResolution.description}")
                }
                
                // 分辨率选择器
                if (showResolutionSelector) {
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 8.dp),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.primaryContainer
                        )
                    ) {
                        Column(
                            modifier = Modifier.padding(16.dp)
                        ) {
                            Text(
                                text = "选择输入分辨率",
                                style = MaterialTheme.typography.titleMedium,
                                modifier = Modifier.padding(bottom = 8.dp)
                            )
                            
                            YOLOPModelManager.Companion.InputResolution.values().forEach { resolution ->
                                Button(
                                    onClick = { 
                                        if (resolution == YOLOPModelManager.Companion.InputResolution.RESOLUTION_320) {
                                            onResolutionChange(resolution)
                                        }
                                    },
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(vertical = 2.dp),
                                    colors = ButtonDefaults.buttonColors(
                                        containerColor = if (resolution == currentResolution)
                                            MaterialTheme.colorScheme.primary
                                        else if (resolution == YOLOPModelManager.Companion.InputResolution.RESOLUTION_320)
                                            MaterialTheme.colorScheme.surface
                                        else MaterialTheme.colorScheme.surfaceVariant
                                    ),
                                    enabled = resolution == YOLOPModelManager.Companion.InputResolution.RESOLUTION_320
                                ) {
                                    Text(
                                        if (resolution == YOLOPModelManager.Companion.InputResolution.RESOLUTION_320) 
                                            resolution.description 
                                        else "${resolution.description} (暂不支持)"
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // 分隔线
        HorizontalDivider(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp),
            thickness = 2.dp,
            color = MaterialTheme.colorScheme.outline
        )
        
        // AI处理结果区域（下半部分）- 显示带标记的实时画面
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .padding(16.dp)
        ) {
            // 显示处理后的图像（带检测标记）
            if (processedBitmap != null) {
                // 使用uiUpdateTrigger强制重新创建AndroidView
                key(uiUpdateTrigger) {
                    AndroidView(
                        factory = { context ->
                            android.widget.ImageView(context).apply {
                                scaleType = android.widget.ImageView.ScaleType.CENTER_CROP
                                setImageBitmap(processedBitmap)
                            }
                        },
                        modifier = Modifier.fillMaxSize(),
                        update = { imageView ->
                            imageView.setImageBitmap(processedBitmap)
                        }
                    )
                }
            } else {
                // 等待处理结果显示
                Card(
                    modifier = Modifier.fillMaxSize(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant
                    )
                ) {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(48.dp),
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                }
            }
            
            // 移除检测结果统计覆盖层，只显示实时视频画面
        }
        
        // 控制按钮区域
        // 移除无用的功能按钮，让界面更简洁
    }
}

/**
 * 检测项目组件
 */
@Composable
fun DetectionItem(icon: String, label: String, count: String) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = icon,
            style = MaterialTheme.typography.headlineMedium
        )
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall
        )
        Text(
            text = count,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.primary
        )
    }
}