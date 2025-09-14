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
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalLifecycleOwner
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
    
    // UI状态管理
    private var previewView: PreviewView? = null
    private var processedBitmap: Bitmap? = null
    
    // 用于触发UI更新的状态
    private var uiUpdateTrigger = 0
    
    // 使用MutableState来触发Compose重新组合
    private val _processedBitmapState = mutableStateOf<Bitmap?>(null)
    private val _uiUpdateTriggerState = mutableStateOf(0)
    
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
                        uiUpdateTrigger = _uiUpdateTriggerState.value
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
     * 开始处理图像
     */
    private fun startProcessing() {
        lifecycleScope.launch {
            var frameCount = 0
            cameraManager.imageFlow.collect { bitmap ->
                frameCount++
                // 每3帧处理一次，平衡性能和流畅度
                if (isModelInitialized && !isProcessing && frameCount % 3 == 0) {
                    isProcessing = true
                    try {
                        processImage(bitmap)
                    } catch (e: Exception) {
                        Log.e(TAG, "处理图像时出错: ${e.message}", e)
                    } finally {
                        isProcessing = false
                    }
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
                    _processedBitmapState.value = renderedBitmap
                    _uiUpdateTriggerState.value = ++uiUpdateTrigger
                    Log.d(TAG, "UI状态已更新 - 触发次数: $uiUpdateTrigger")
                }
                
                Log.d(TAG, "处理完成 - FPS: ${result.fps}, 检测数量: ${result.detections.size}")
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
    
    override fun onDestroy() {
        super.onDestroy()
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
    uiUpdateTrigger: Int
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
        
        // 主摄像头预览区域（上半部分）
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .padding(horizontal = 16.dp)
        ) {
            // 摄像头预览
            if (previewView != null) {
                AndroidView(
                    factory = { previewView!! },
                    modifier = Modifier.fillMaxSize()
                )
            } else {
                // 加载中显示
                Box(
                    modifier = Modifier.fillMaxSize(),
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