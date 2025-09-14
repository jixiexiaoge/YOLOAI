package com.example.yoloai.camera

import android.content.Context
import android.graphics.*
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import android.util.Size
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.receiveAsFlow

/**
 * CameraX管理器
 * 负责摄像头初始化和图像捕获
 * 
 * 主要功能：
 * 1. 初始化摄像头
 * 2. 配置预览和图像分析
 * 3. 提供图像数据流
 * 4. 处理图像格式转换
 */
class CameraManager(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner
) {
    
    companion object {
        private const val TAG = "CameraManager"
        private const val TARGET_ASPECT_RATIO = AspectRatio.RATIO_16_9
        private const val PREVIEW_WIDTH = 1280
        private const val PREVIEW_HEIGHT = 720
    }
    
    private var cameraProvider: ProcessCameraProvider? = null
    private var camera: androidx.camera.core.Camera? = null
    private var imageAnalyzer: ImageAnalysis? = null
    
    // 图像数据通道
    private val imageChannel = Channel<Bitmap>(capacity = 2)
    val imageFlow = imageChannel.receiveAsFlow()
    
    /**
     * 初始化摄像头
     */
    suspend fun initialize(previewView: PreviewView): Boolean {
        return try {
            // 获取CameraProvider
            val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
            cameraProvider = cameraProviderFuture.get()
            
            // 配置预览 - 使用宽高比而不是固定分辨率
            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .build()
            
            // 配置图像分析 - 使用320x320分辨率用于AI处理
            imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(Size(320, 320)) // 使用320x320分辨率用于AI处理
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { analyzer ->
                    analyzer.setAnalyzer(
                        ContextCompat.getMainExecutor(context),
                        YOLOPImageAnalyzer { bitmap ->
                            // 将图像发送到通道
                            if (!imageChannel.isClosedForSend) {
                                imageChannel.trySend(bitmap)
                            }
                        }
                    )
                }
            
            // 绑定到生命周期
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            
            cameraProvider?.unbindAll()
            camera = cameraProvider?.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview,
                imageAnalyzer
            )
            
            // 设置预览
            preview.setSurfaceProvider(previewView.surfaceProvider)
            
            Log.i(TAG, "摄像头初始化成功")
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "摄像头初始化失败: ${e.message}", e)
            false
        }
    }
    
    /**
     * 释放资源
     */
    fun release() {
        try {
            imageChannel.close()
            cameraProvider?.unbindAll()
            Log.i(TAG, "摄像头资源已释放")
        } catch (e: Exception) {
            Log.e(TAG, "释放摄像头资源时出错: ${e.message}", e)
        }
    }
    
    /**
     * 获取摄像头信息
     */
    fun getCameraInfo(): androidx.camera.core.CameraInfo? {
        return camera?.cameraInfo
    }
}

/**
 * 图像分析器
 * 将CameraX的Image转换为Bitmap
 */
private class YOLOPImageAnalyzer(
    private val onImageCaptured: (Bitmap) -> Unit
) : ImageAnalysis.Analyzer {
    
    companion object {
        private const val TAG = "YOLOPImageAnalyzer"
    }
    
    override fun analyze(imageProxy: ImageProxy) {
        try {
            val bitmap = imageProxyToBitmap(imageProxy)
            if (bitmap != null) {
                onImageCaptured(bitmap)
            }
        } catch (e: Exception) {
            Log.e(TAG, "图像分析失败: ${e.message}", e)
        } finally {
            imageProxy.close()
        }
    }
    
    /**
     * 将ImageProxy转换为Bitmap
     */
    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return when (imageProxy.format) {
            ImageFormat.YUV_420_888 -> {
                yuv420ToBitmap(imageProxy)
            }
            ImageFormat.NV21 -> {
                nv21ToBitmap(imageProxy)
            }
            else -> {
                Log.w(TAG, "不支持的图像格式: ${imageProxy.format}")
                null
            }
        }
    }
    
    /**
     * YUV420_888格式转Bitmap
     */
    private fun yuv420ToBitmap(imageProxy: ImageProxy): Bitmap? {
        val planes = imageProxy.planes
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer
        
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        
        val nv21 = ByteArray(ySize + uSize + vSize)
        
        // Y平面
        yBuffer.get(nv21, 0, ySize)
        
        // UV平面交错排列
        val uvPixelStride = planes[1].pixelStride
        if (uvPixelStride == 1) {
            // 标准格式
            uBuffer.get(nv21, ySize, uSize)
            vBuffer.get(nv21, ySize + uSize, vSize)
        } else {
            // 交错格式
            val uvBuffer = ByteBuffer.allocate(uSize + vSize)
            val uArray = ByteArray(uSize)
            val vArray = ByteArray(vSize)
            uBuffer.get(uArray)
            vBuffer.get(vArray)
            
            for (i in 0 until uSize) {
                uvBuffer.put(uArray[i])
                uvBuffer.put(vArray[i])
            }
            
            System.arraycopy(uvBuffer.array(), 0, nv21, ySize, uvBuffer.array().size)
        }
        
        return nv21ToBitmap(nv21, imageProxy.width, imageProxy.height)
    }
    
    /**
     * NV21格式转Bitmap
     */
    private fun nv21ToBitmap(imageProxy: ImageProxy): Bitmap? {
        val buffer = imageProxy.planes[0].buffer
        val nv21 = ByteArray(buffer.remaining())
        buffer.get(nv21)
        return nv21ToBitmap(nv21, imageProxy.width, imageProxy.height)
    }
    
    /**
     * NV21字节数组转Bitmap
     */
    private fun nv21ToBitmap(nv21: ByteArray, width: Int, height: Int): Bitmap? {
        return try {
            val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
            val imageBytes = out.toByteArray()
            
            // 这里需要将JPEG字节数组转换为Bitmap
            // 为了简化，我们使用一个更直接的方法
            convertNV21ToBitmap(nv21, width, height)
            
        } catch (e: Exception) {
            Log.e(TAG, "NV21转Bitmap失败: ${e.message}", e)
            null
        }
    }
    
    /**
     * 直接转换NV21到Bitmap（优化版本）
     */
    private fun convertNV21ToBitmap(nv21: ByteArray, width: Int, height: Int): Bitmap? {
        return try {
            // 使用更高效的转换方法
            val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, width, height), 80, out) // 降低质量以减少处理时间
            val imageBytes = out.toByteArray()
            
            // 将字节数组转换为Bitmap
            val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            
            // 如果转换失败，使用备用方法
            if (bitmap == null) {
                Log.w(TAG, "YuvImage转换失败，使用备用方法")
                return createFallbackBitmap(width, height)
            }
            
            // 修复旋转问题：将图像旋转90度
            val rotatedBitmap = rotateBitmap(bitmap, 90f)
            bitmap.recycle() // 释放原始bitmap
            
            rotatedBitmap
            
        } catch (e: Exception) {
            Log.e(TAG, "NV21转换失败: ${e.message}", e)
            createFallbackBitmap(width, height)
        }
    }
    
    /**
     * 旋转Bitmap
     */
    private fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(degrees)
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
    
    /**
     * 创建备用Bitmap（用于测试）
     */
    private fun createFallbackBitmap(width: Int, height: Int): Bitmap? {
        return try {
            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(bitmap)
            val paint = Paint().apply {
                color = Color.GRAY
                style = Paint.Style.FILL
            }
            canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), paint)
            bitmap
        } catch (e: Exception) {
            Log.e(TAG, "创建备用Bitmap失败: ${e.message}", e)
            null
        }
    }
}
