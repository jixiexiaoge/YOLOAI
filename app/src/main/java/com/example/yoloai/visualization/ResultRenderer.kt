package com.example.yoloai.visualization

import android.graphics.*
import android.util.Log
import com.example.yoloai.onnx.Detection
import com.example.yoloai.onnx.YOLOPResult
import kotlin.math.max
import kotlin.math.min

/**
 * 结果渲染器
 * 负责在图像上绘制检测和分割结果
 * 
 * 主要功能：
 * 1. 绘制车辆检测框
 * 2. 绘制车道线（实线/虚线）
 * 3. 绘制可行驶区域
 * 4. 显示FPS信息
 */
class ResultRenderer {
    
    companion object {
        private const val TAG = "ResultRenderer"
        
        // 颜色定义 - 严格按照demo copy.py的BGR格式
        private val VEHICLE_COLOR = Color.RED          // 车辆框保持红色
        private val SOLID_LANE_COLOR = Color.BLUE      // 实线用蓝色 (BGR: 255,0,0)
        private val DASHED_LANE_COLOR = Color.YELLOW   // 虚线用黄色 (BGR: 0,255,255)
        private val DRIVABLE_AREA_COLOR = Color.argb(128, 0, 255, 0) // 半透明绿色 (BGR: 0,255,0, alpha=0.5)
        private val TEXT_COLOR = Color.WHITE
        private val TEXT_BACKGROUND_COLOR = Color.argb(150, 0, 0, 0)
        
        // 绘制参数 - 优化版本
        private const val BOX_THICKNESS = 2f // 减少线条粗细
        private const val LANE_THICKNESS = 3f
        private const val TEXT_SIZE = 20f // 减少字体大小
        private const val TEXT_PADDING = 6f
        private const val FPS_TEXT_SIZE = 28f
        
        // 性能优化参数
        private const val MAX_DETECTION_BOXES = 20 // 限制检测框数量
        private const val SKIP_FRAME_RENDER = 2 // 每2帧渲染一次UI
    }
    
    private val paint = Paint().apply {
        isAntiAlias = true
        style = Paint.Style.STROKE
        strokeWidth = BOX_THICKNESS
    }
    
    private val textPaint = Paint().apply {
        isAntiAlias = true
        color = TEXT_COLOR
        textSize = TEXT_SIZE
        typeface = Typeface.DEFAULT_BOLD
    }
    
    private val fpsPaint = Paint().apply {
        isAntiAlias = true
        color = TEXT_COLOR
        textSize = FPS_TEXT_SIZE
        typeface = Typeface.DEFAULT_BOLD
    }
    
    private val backgroundPaint = Paint().apply {
        color = TEXT_BACKGROUND_COLOR
        style = Paint.Style.FILL
    }
    
    /**
     * 在图像上绘制检测结果
     * @param bitmap 原始图像
     * @param result YOLOP推理结果
     * @return 绘制后的图像
     */
    fun renderResults(bitmap: Bitmap, result: YOLOPResult): Bitmap {
        return try {
            val outputBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
            val canvas = Canvas(outputBitmap)
            
            // 1. 绘制可行驶区域（底层）
            renderDrivableArea(canvas, result.daSegMask, bitmap.width, bitmap.height)
            
            // 2. 绘制车道线（中层）
            renderLaneLines(canvas, result.llSegMask, bitmap.width, bitmap.height)
            
            // 3. 绘制车辆检测框（顶层）
            renderDetections(canvas, result.detections)
            
            // 4. 绘制FPS信息（最顶层）
            renderFPSInfo(canvas, result.fps, result.inferenceTime)
            
            outputBitmap
            
        } catch (e: Exception) {
            Log.e(TAG, "渲染结果失败: ${e.message}", e)
            bitmap
        }
    }
    
    /**
     * 绘制可行驶区域 - 使用像素级渲染（仿照Python代码）
     */
    private fun renderDrivableArea(
        canvas: Canvas,
        daSegMask: Array<IntArray>?,
        width: Int,
        height: Int
    ) {
        if (daSegMask == null) return
        
        // 将320x320的分割掩码缩放到实际图像尺寸
        val scaleX = width.toFloat() / daSegMask[0].size
        val scaleY = height.toFloat() / daSegMask.size
        
        // 使用像素级绘制，仿照Python的show_seg_result函数
        val areaPaint = Paint().apply {
            color = Color.GREEN // 可行驶区域用绿色，仿照Python代码
            alpha = 128 // 半透明效果
            style = Paint.Style.FILL
        }
        
        // 优化：使用批量绘制减少Canvas调用次数
        val path = Path()
        var pathStarted = false
        var batchCount = 0
        val maxBatchSize = 1000 // 限制批处理大小避免路径过大
        
        // 遍历分割掩码，绘制可行驶区域像素
        for (y in daSegMask.indices) {
            for (x in daSegMask[y].indices) {
                if (daSegMask[y][x] == 1) { // 可行驶区域像素
                    val screenX = x * scaleX
                    val screenY = y * scaleY
                    val pixelWidth = maxOf(1f, scaleX)
                    val pixelHeight = maxOf(1f, scaleY)
                    
                    // 添加矩形到路径
                    if (!pathStarted) {
                        path.moveTo(screenX, screenY)
                        pathStarted = true
                    }
                    path.addRect(
                        screenX, screenY,
                        screenX + pixelWidth, screenY + pixelHeight,
                        Path.Direction.CW
                    )
                    
                    batchCount++
                    // 当批次达到最大大小时，绘制并重置
                    if (batchCount >= maxBatchSize) {
                        canvas.drawPath(path, areaPaint)
                        path.reset()
                        pathStarted = false
                        batchCount = 0
                    }
                }
            }
        }
        
        // 绘制剩余的路径
        if (pathStarted) {
            canvas.drawPath(path, areaPaint)
        }
    }
    
    
    /**
     * 绘制车道线 - 区分实线和虚线，使用不同颜色
     */
    private fun renderLaneLines(
        canvas: Canvas,
        llSegMask: Array<IntArray>?,
        width: Int,
        height: Int
    ) {
        if (llSegMask == null) return
        
        // 将320x320的分割掩码缩放到实际图像尺寸
        val scaleX = width.toFloat() / llSegMask[0].size
        val scaleY = height.toFloat() / llSegMask.size
        
        // 使用像素级绘制，区分实线和虚线
        paint.style = Paint.Style.FILL
        
        // 为实线和虚线分别创建路径
        val solidPath = Path()
        val dashedPath = Path()
        var solidPathStarted = false
        var dashedPathStarted = false
        var solidBatchCount = 0
        var dashedBatchCount = 0
        val maxBatchSize = 500 // 车道线批处理大小
        
        // 遍历分割掩码，绘制车道线像素
        for (y in llSegMask.indices) {
            for (x in llSegMask[y].indices) {
                val pixelValue = llSegMask[y][x]
                if (pixelValue > 0) { // 车道线像素
                    val screenX = x * scaleX
                    val screenY = y * scaleY
                    val pixelWidth = maxOf(1f, scaleX)
                    val pixelHeight = maxOf(1f, scaleY)
                    
                    // 根据像素值添加到不同路径
                    when (pixelValue) {
                        1 -> { // 实线
                            if (!solidPathStarted) {
                                solidPath.moveTo(screenX, screenY)
                                solidPathStarted = true
                            }
                            solidPath.addRect(
                                screenX, screenY,
                                screenX + pixelWidth, screenY + pixelHeight,
                                Path.Direction.CW
                            )
                            solidBatchCount++
                            // 批处理优化
                            if (solidBatchCount >= maxBatchSize) {
                                paint.color = Color.BLUE
                                canvas.drawPath(solidPath, paint)
                                solidPath.reset()
                                solidPathStarted = false
                                solidBatchCount = 0
                            }
                        }
                        2 -> { // 虚线
                            if (!dashedPathStarted) {
                                dashedPath.moveTo(screenX, screenY)
                                dashedPathStarted = true
                            }
                            dashedPath.addRect(
                                screenX, screenY,
                                screenX + pixelWidth, screenY + pixelHeight,
                                Path.Direction.CW
                            )
                            dashedBatchCount++
                            // 批处理优化
                            if (dashedBatchCount >= maxBatchSize) {
                                paint.color = Color.YELLOW
                                canvas.drawPath(dashedPath, paint)
                                dashedPath.reset()
                                dashedPathStarted = false
                                dashedBatchCount = 0
                            }
                        }
                    }
                }
            }
        }
        
        // 一次性绘制所有实线
        if (solidPathStarted) {
            paint.color = Color.BLUE
            canvas.drawPath(solidPath, paint)
        }
        
        // 一次性绘制所有虚线
        if (dashedPathStarted) {
            paint.color = Color.YELLOW
            canvas.drawPath(dashedPath, paint)
        }
    }
    
    
    
    
    
    /**
     * 绘制检测框 - 使用紫色框标记车辆
     */
    private fun renderDetections(canvas: Canvas, detections: List<Detection>) {
        paint.color = VEHICLE_COLOR
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = BOX_THICKNESS
        
        // 限制检测框数量以提高性能
        val limitedDetections = detections.take(MAX_DETECTION_BOXES)
        
        for (detection in limitedDetections) {
            // 绘制检测框
            val rect = RectF(
                detection.x1,
                detection.y1,
                detection.x2,
                detection.y2
            )
            canvas.drawRect(rect, paint)
            
            // 只绘制高置信度的标签以节省性能
            if (detection.confidence > 0.6f) {
                val label = "${detection.className} ${String.format("%.0f%%", detection.confidence * 100)}"
                val textBounds = Rect()
                textPaint.getTextBounds(label, 0, label.length, textBounds)
                
                val textX = detection.x1
                val textY = detection.y1 - TEXT_PADDING
                
                // 绘制文本背景
                val backgroundRect = RectF(
                    textX,
                    textY - textBounds.height() - TEXT_PADDING,
                    textX + textBounds.width() + TEXT_PADDING * 2,
                    textY + TEXT_PADDING
                )
                canvas.drawRect(backgroundRect, backgroundPaint)
                
                // 绘制文本
                canvas.drawText(label, textX + TEXT_PADDING, textY, textPaint)
            }
        }
    }
    
    /**
     * 绘制FPS信息和图例
     */
    private fun renderFPSInfo(canvas: Canvas, fps: Float, inferenceTime: Long) {
        val fpsText = "FPS: ${String.format("%.1f", fps)}"
        val timeText = "Time: ${inferenceTime}ms"
        
        val x = 20f
        var y = 30f
        
        // 绘制图例
        val legendPaint = Paint().apply {
            isAntiAlias = true
            color = TEXT_COLOR
            textSize = 24f
            typeface = Typeface.DEFAULT_BOLD
        }
        
        // 图例背景
        val legendBackground = RectF(
            x - 10f,
            y - 20f,
            x + 300f,
            y + 125f
        )
        canvas.drawRect(legendBackground, backgroundPaint)
        
        // 绘制图例文字 - 严格按照demo copy.py
        canvas.drawText("Blue: Solid Lines", x, y, legendPaint)
        y += 25f
        canvas.drawText("Yellow: Dashed Lines", x, y, legendPaint)
        y += 25f
        canvas.drawText("Green: Drivable Area", x, y, legendPaint)
        y += 25f
        canvas.drawText("Red: Vehicle", x, y, legendPaint)
        
        // 绘制FPS信息
        y += 40f
        canvas.drawText(fpsText, x, y, fpsPaint)
        y += 30f
        canvas.drawText(timeText, x, y, fpsPaint)
    }
}

/**
 * 点数据类
 */
data class Point(val x: Int, val y: Int)

