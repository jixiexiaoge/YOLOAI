package com.example.yoloai.onnx

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min
import ai.onnxruntime.*

/**
 * YOLOP模型管理器
 * 负责加载ONNX模型并执行真实的AI推理
 * 
 * 功能：
 * 1. 加载YOLOP ONNX模型
 * 2. 预处理摄像头画面
 * 3. 执行模型推理
 * 4. 后处理推理结果
 */
class YOLOPModelManager(private val context: Context) {
    
    companion object {
        private const val TAG = "YOLOPModelManager"
        private const val MODEL_NAME = "yolop-320-320.onnx"
        private const val INPUT_SIZE = 320
        
        // 支持多种分辨率优化
        enum class InputResolution(val size: Int, val description: String) {
            RESOLUTION_256(256, "256x256 - 最高性能"),
            RESOLUTION_288(288, "288x288 - 平衡性能"),
            RESOLUTION_320(320, "320x320 - 原始精度")
        }
        
        // 当前使用的分辨率，可以根据性能需求调整
        // 注意：由于模型文件是320x320固定版本，暂时只能使用320分辨率
        private var currentResolution = InputResolution.RESOLUTION_320
        private const val CONFIDENCE_THRESHOLD = 0.05f // 进一步降低置信度阈值，提高检测敏感度
        private const val IOU_THRESHOLD = 0.45f
        
        // 性能优化参数
        private const val ENABLE_DETAILED_SEGMENTATION = false // 暂时禁用详细分割处理
        private const val SEGMENTATION_SAMPLE_RATE = 4 // 每4个像素采样一次，减少计算量
        
        // YOLOP模型输入输出层名称（根据export_onnx.py）
        private const val INPUT_NAME = "images"
        private const val DETECTION_OUTPUT = "det_out"
        private const val DA_SEG_OUTPUT = "drive_area_seg"
        private const val LL_SEG_OUTPUT = "lane_line_seg"
    }
    
    private var isModelLoaded = false
    private var onnxSession: OrtSession? = null
    private var onnxEnvironment: OrtEnvironment? = null
    private var modelPath: String? = null
    
    /**
     * 初始化ONNX Runtime环境并加载模型
     */
    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        try {
            // 1. 检查模型文件是否存在
            val modelExists = try {
                context.assets.open(MODEL_NAME).use { true }
            } catch (e: Exception) {
                Log.e(TAG, "模型文件不存在: ${e.message}")
                false
            }
            
            if (!modelExists) {
                Log.e(TAG, "模型文件 $MODEL_NAME 不存在于assets目录")
                return@withContext false
            }
            
            // 2. 将模型文件从assets复制到内部存储
            modelPath = copyModelFromAssets()
            if (modelPath == null) {
                Log.e(TAG, "复制模型文件失败")
                return@withContext false
            }
            
            // 3. 初始化ONNX Runtime
            if (!initializeONNXRuntime()) {
                Log.e(TAG, "ONNX Runtime初始化失败")
                return@withContext false
            }
            
            // 4. 加载模型
            if (!loadONNXModel()) {
                Log.e(TAG, "ONNX模型加载失败")
                return@withContext false
            }
            
            isModelLoaded = true
            Log.i(TAG, "YOLOP模型加载成功")
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "模型加载失败: ${e.message}", e)
            isModelLoaded = false
            false
        }
    }
    
    /**
     * 从assets复制模型文件到内部存储
     */
    private fun copyModelFromAssets(): String? {
        return try {
            val internalDir = File(context.filesDir, "models")
            if (!internalDir.exists()) {
                internalDir.mkdirs()
            }
            
            val modelFile = File(internalDir, MODEL_NAME)
            if (modelFile.exists()) {
                Log.i(TAG, "模型文件已存在: ${modelFile.absolutePath}")
                return modelFile.absolutePath
            }
            
            context.assets.open(MODEL_NAME).use { inputStream ->
                FileOutputStream(modelFile).use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
            
            Log.i(TAG, "模型文件复制成功: ${modelFile.absolutePath}")
            modelFile.absolutePath
        } catch (e: Exception) {
            Log.e(TAG, "复制模型文件失败: ${e.message}", e)
            null
        }
    }
    
    /**
     * 初始化ONNX Runtime
     */
    private fun initializeONNXRuntime(): Boolean {
        return try {
            onnxEnvironment = OrtEnvironment.getEnvironment()
            Log.i(TAG, "ONNX Runtime初始化成功")
            true
        } catch (e: Exception) {
            Log.e(TAG, "ONNX Runtime初始化失败: ${e.message}", e)
            false
        }
    }
    
    /**
     * 加载ONNX模型 - 智能配置和回退机制
     */
    private fun loadONNXModel(): Boolean {
        return try {
            if (onnxEnvironment == null || modelPath == null) {
                Log.e(TAG, "ONNX环境或模型路径为空")
                return false
            }
            
            // 尝试多种配置策略
            val configurations = listOf(
                createNNAPIConfig(),
                createOptimizedCPUConfig(),
                createBasicCPUConfig()
            )
            
            for ((index, config) in configurations.withIndex()) {
                try {
                    Log.i(TAG, "尝试配置 ${index + 1}: ${config.description}")
                    onnxSession = onnxEnvironment!!.createSession(modelPath!!, config.options)
                    Log.i(TAG, "模型加载成功: ${config.description}")
                    return true
                } catch (e: Exception) {
                    Log.w(TAG, "配置 ${index + 1} 失败: ${e.message}")
                    if (index == configurations.size - 1) {
                        Log.e(TAG, "所有配置都失败，模型加载失败")
                        throw e
                    }
                }
            }
            
            false
        } catch (e: Exception) {
            Log.e(TAG, "ONNX模型加载失败: ${e.message}", e)
            false
        }
    }
    
    /**
     * 创建NNAPI配置
     */
    private fun createNNAPIConfig(): ConfigResult {
        val sessionOptions = OrtSession.SessionOptions()
        sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        
        try {
            sessionOptions.addNnapi()
            sessionOptions.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL)
            sessionOptions.setInterOpNumThreads(4)
            sessionOptions.setIntraOpNumThreads(4)
            sessionOptions.setMemoryPatternOptimization(true)
            return ConfigResult(sessionOptions, "NNAPI硬件加速配置")
        } catch (e: Exception) {
            throw Exception("NNAPI配置失败: ${e.message}")
        }
    }
    
    /**
     * 创建优化的CPU配置
     */
    private fun createOptimizedCPUConfig(): ConfigResult {
        val sessionOptions = OrtSession.SessionOptions()
        sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        sessionOptions.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL)
        sessionOptions.setInterOpNumThreads(6) // 增加线程数
        sessionOptions.setIntraOpNumThreads(6) // 增加线程数
        try {
            sessionOptions.setMemoryPatternOptimization(true)
        } catch (e: Exception) {
            Log.w(TAG, "内存优化不可用")
        }
        return ConfigResult(sessionOptions, "高度优化CPU配置")
    }
    
    /**
     * 创建基础CPU配置
     */
    private fun createBasicCPUConfig(): ConfigResult {
        val sessionOptions = OrtSession.SessionOptions()
        sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
        sessionOptions.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
        sessionOptions.setInterOpNumThreads(1)
        sessionOptions.setIntraOpNumThreads(1)
        return ConfigResult(sessionOptions, "基础CPU配置")
    }
    
    /**
     * 配置结果数据类
     */
    private data class ConfigResult(
        val options: OrtSession.SessionOptions,
        val description: String
    )
    
    /**
     * 加载ONNX模型 - 原始方法（已废弃）
     */
    private fun loadONNXModelOld(): Boolean {
        return try {
            if (onnxEnvironment == null || modelPath == null) {
                Log.e(TAG, "ONNX环境或模型路径为空")
                return false
            }
            
            val sessionOptions = OrtSession.SessionOptions()
            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            
            // 智能启用NNAPI硬件加速 - 包含回退机制
            var nnapiEnabled = false
            try {
                // 尝试启用NNAPI
                sessionOptions.addNnapi()
                nnapiEnabled = true
                Log.i(TAG, "NNAPI硬件加速已启用")
            } catch (e: Exception) {
                Log.w(TAG, "NNAPI不可用，使用CPU推理: ${e.message}")
                // 如果NNAPI失败，继续使用CPU推理
            }
            
            // 根据NNAPI状态优化配置
            if (nnapiEnabled) {
                // NNAPI模式下使用优化配置
                sessionOptions.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL)
                sessionOptions.setInterOpNumThreads(4) // NNAPI可以使用更多线程
                sessionOptions.setIntraOpNumThreads(4)
                Log.i(TAG, "使用NNAPI优化配置：并行执行，4线程")
            } else {
                // CPU模式下使用保守配置
                sessionOptions.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
                sessionOptions.setInterOpNumThreads(2) // CPU模式使用较少线程
                sessionOptions.setIntraOpNumThreads(2)
                Log.i(TAG, "使用CPU保守配置：顺序执行，2线程")
            }
            
            // 启用内存优化
            try {
                sessionOptions.setMemoryPatternOptimization(true)
                Log.i(TAG, "内存模式优化已启用")
            } catch (e: Exception) {
                Log.w(TAG, "内存模式优化不可用: ${e.message}")
            }
            
            onnxSession = onnxEnvironment!!.createSession(modelPath!!, sessionOptions)
            
            // 打印模型信息用于调试
            val inputNames = onnxSession!!.inputNames
            val outputNames = onnxSession!!.outputNames
            Log.i(TAG, "模型输入层: ${inputNames.joinToString()}")
            Log.i(TAG, "模型输出层: ${outputNames.joinToString()}")
            
            Log.i(TAG, "ONNX模型加载成功: $modelPath")
            true
        } catch (e: Exception) {
            Log.e(TAG, "ONNX模型加载失败: ${e.message}", e)
            false
        }
    }
    
    // 性能监控变量
    private var totalInferenceTime = 0L
    private var inferenceCount = 0
    private var lastPerformanceLogTime = System.currentTimeMillis()
    
    /**
     * 执行推理 - 优化的AI推理
     * @param bitmap 输入图像
     * @return 推理结果
     */
    suspend fun inference(bitmap: Bitmap): YOLOPResult? = withContext(Dispatchers.IO) {
        if (!isModelLoaded) {
            Log.e(TAG, "模型未加载")
            return@withContext null
        }
        
        try {
            val startTime = System.currentTimeMillis()
            
            // 1. 预处理图像
            val preprocessStart = System.currentTimeMillis()
            val inputTensor = preprocessImage(bitmap)
            if (inputTensor == null) {
                Log.e(TAG, "图像预处理失败")
                return@withContext null
            }
            val preprocessTime = System.currentTimeMillis() - preprocessStart
            
            // 2. 执行模型推理
            val inferenceStart = System.currentTimeMillis()
            val outputs = runInference(inputTensor)
            if (outputs == null) {
                Log.e(TAG, "模型推理失败")
                return@withContext null
            }
            val inferenceTime = System.currentTimeMillis() - inferenceStart
            
            // 3. 后处理结果
            val postprocessStart = System.currentTimeMillis()
            val detections = postprocessDetections(outputs.detectionOutput, bitmap.width, bitmap.height)
            val daSegMask = postprocessSegmentation(outputs.daSegOutput)
            val llSegMask = postprocessLaneLineSegmentation(outputs.llSegOutput)
            val postprocessTime = System.currentTimeMillis() - postprocessStart
            
            val totalTime = System.currentTimeMillis() - startTime
            
            // 性能统计
            totalInferenceTime += totalTime
            inferenceCount++
            
            // 每10次推理输出一次性能统计
            if (inferenceCount % 10 == 0) {
                val avgTime = totalInferenceTime / inferenceCount
                val currentTime = System.currentTimeMillis()
                if (currentTime - lastPerformanceLogTime > 5000) { // 每5秒输出一次
                    Log.i(TAG, "性能统计 - 平均推理时间: ${avgTime}ms, 平均FPS: ${String.format("%.1f", 1000.0f / avgTime)}, 总推理次数: $inferenceCount")
                    lastPerformanceLogTime = currentTime
                }
            }
            
            val result = YOLOPResult(
                detections = detections,
                daSegMask = daSegMask,
                llSegMask = llSegMask,
                inferenceTime = totalTime,
                fps = 1000.0f / totalTime
            )
            
            Log.d(TAG, "推理完成 - 总时间: ${totalTime}ms (预处理: ${preprocessTime}ms, 推理: ${inferenceTime}ms, 后处理: ${postprocessTime}ms), 检测数量: ${detections.size}")
            result
            
        } catch (e: Exception) {
            Log.e(TAG, "推理失败: ${e.message}", e)
            null
        }
    }
    
    // 预分配的缓冲区，避免重复内存分配
    private var preprocessBuffer: FloatBuffer? = null
    private var outputBuffers: MutableMap<String, FloatArray> = mutableMapOf()
    private var inputTensor: OnnxTensor? = null
    
    /**
     * 预处理图像 - 将Bitmap转换为模型输入格式（优化版本）
     */
    private fun preprocessImage(bitmap: Bitmap): FloatBuffer? {
        return try {
            // 1. 调整图像大小到当前分辨率
            val targetSize = currentResolution.size
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, targetSize, targetSize, true)
            
            // 2. 转换为RGB格式
            val rgbBitmap = if (resizedBitmap.config == Bitmap.Config.ARGB_8888) {
                resizedBitmap
            } else {
                resizedBitmap.copy(Bitmap.Config.ARGB_8888, false)
            }
            
            // 3. 归一化到[0,1]范围并转换为CHW格式
            val normalizedData = normalizeImageOptimized(rgbBitmap)
            
            // 4. 使用预分配的缓冲区，避免重复内存分配
            if (preprocessBuffer == null) {
                preprocessBuffer = ByteBuffer.allocateDirect(normalizedData.size * 4)
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer()
            }
            
            preprocessBuffer!!.rewind()
            preprocessBuffer!!.put(normalizedData)
            preprocessBuffer!!.rewind()
            
            Log.d(TAG, "图像预处理完成，尺寸: ${rgbBitmap.width}x${rgbBitmap.height}")
            preprocessBuffer
            
        } catch (e: Exception) {
            Log.e(TAG, "图像预处理失败: ${e.message}", e)
            null
        }
    }
    
    /**
     * 归一化图像数据并转换为CHW格式（优化版本）
     */
    private fun normalizeImageOptimized(bitmap: Bitmap): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        
        val normalizedData = FloatArray(width * height * 3)
        val wh = width * height
        
        // 使用索引计算优化循环
        for (i in pixels.indices) {
            val pixel = pixels[i]
            // 使用位运算提取RGB值并归一化
            normalizedData[i] = ((pixel shr 16) and 0xFF) / 255.0f
            normalizedData[wh + i] = ((pixel shr 8) and 0xFF) / 255.0f
            normalizedData[2 * wh + i] = (pixel and 0xFF) / 255.0f
        }
        
        return normalizedData
    }
    
    /**
     * 执行模型推理 - 使用真正的ONNX Runtime
     */
    private fun runInference(inputTensor: FloatBuffer): ModelOutputs? {
        return try {
            if (onnxSession == null) {
                Log.e(TAG, "ONNX会话未初始化")
                return null
            }
            
            Log.d(TAG, "执行真实的ONNX模型推理")
            
            // 创建输入张量 - 使用动态分辨率，复用预分配的tensor
            val targetSize = currentResolution.size
            val inputShape = longArrayOf(1, 3, targetSize.toLong(), targetSize.toLong())
            
            // 创建输入张量 - 简化版本，避免复杂的tensor复用
            val inputOnnxTensor = OnnxTensor.createTensor(onnxEnvironment!!, inputTensor, inputShape)
            
            // 准备输入映射
            val inputs = mapOf(INPUT_NAME to inputOnnxTensor)
            
            // 执行推理
            val results = onnxSession!!.run(inputs)
            
            // 打印可用的输出层名称用于调试
            Log.d(TAG, "推理结果数量: ${results.size()}")
            
            // 提取输出 - 使用正确的ONNX Runtime API
            val detectionOptional = results.get(DETECTION_OUTPUT)
            val daSegOptional = results.get(DA_SEG_OUTPUT)
            val llSegOptional = results.get(LL_SEG_OUTPUT)
            
            if (!detectionOptional.isPresent || !daSegOptional.isPresent || !llSegOptional.isPresent) {
                Log.e(TAG, "输出张量不存在")
                inputOnnxTensor.close()
                results.close()
                return null
            }
            
            val detectionTensor = detectionOptional.get() as OnnxTensor
            val daSegTensor = daSegOptional.get() as OnnxTensor
            val llSegTensor = llSegOptional.get() as OnnxTensor
            
            // 直接使用缓冲区而不是复制到数组
            val detectionOutput = detectionTensor.floatBuffer
            val daSegOutput = daSegTensor.floatBuffer
            val llSegOutput = llSegTensor.floatBuffer
            
            // 创建输出数组
            val detectionArray = FloatArray(detectionOutput.remaining())
            val daSegArray = FloatArray(daSegOutput.remaining())
            val llSegArray = FloatArray(llSegOutput.remaining())
            
            // 保存当前位置
            val detectionPos = detectionOutput.position()
            val daSegPos = daSegOutput.position()
            val llSegPos = llSegOutput.position()
            
            // 获取数据
            detectionOutput.get(detectionArray)
            daSegOutput.get(daSegArray)
            llSegOutput.get(llSegArray)
            
            // 恢复位置
            detectionOutput.position(detectionPos)
            daSegOutput.position(daSegPos)
            llSegOutput.position(llSegPos)
            
            // 清理资源
            inputOnnxTensor.close()
            results.close()
            
            Log.d(TAG, "ONNX推理完成")
            Log.d(TAG, "检测输出形状: ${detectionArray.size}")
            Log.d(TAG, "可行驶区域输出形状: ${daSegArray.size}")
            Log.d(TAG, "车道线输出形状: ${llSegArray.size}")
            
            ModelOutputs(
                detectionOutput = detectionArray,
                daSegOutput = daSegArray,
                llSegOutput = llSegArray
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "ONNX模型推理失败: ${e.message}", e)
            null
        }
    }
    
    /**
     * 后处理检测结果 - 实现真正的NMS
     */
    private fun postprocessDetections(
        detectionOutput: FloatArray,
        originalWidth: Int,
        originalHeight: Int
    ): List<Detection> {
        val detections = mutableListOf<Detection>()
        
        try {
            // YOLOP检测输出格式: [batch, num_detections, 6]
            // 6 = [x_center, y_center, width, height, confidence, class_id]
            val numDetections = detectionOutput.size / 6
            
            val candidates = mutableListOf<Detection>()
            
            // 解析检测结果
            for (i in 0 until numDetections) {
                val baseIndex = i * 6
                val xCenter = detectionOutput[baseIndex]
                val yCenter = detectionOutput[baseIndex + 1]
                val width = detectionOutput[baseIndex + 2]
                val height = detectionOutput[baseIndex + 3]
                val confidence = detectionOutput[baseIndex + 4]
                val classId = detectionOutput[baseIndex + 5].toInt()
                
                // 过滤低置信度检测
                if (confidence >= CONFIDENCE_THRESHOLD) {
                    // 转换为边界框格式
                    val x1 = (xCenter - width / 2) * originalWidth / INPUT_SIZE
                    val y1 = (yCenter - height / 2) * originalHeight / INPUT_SIZE
                    val x2 = (xCenter + width / 2) * originalWidth / INPUT_SIZE
                    val y2 = (yCenter + height / 2) * originalHeight / INPUT_SIZE
                    
                    candidates.add(
                Detection(
                    x1 = x1,
                    y1 = y1,
                    x2 = x2,
                    y2 = y2,
                    confidence = confidence,
                    classId = classId,
                    className = getClassName(classId)
                )
            )
                }
            }
            
            // 应用NMS
            detections.addAll(applyNMSOptimized(candidates))
            
            Log.d(TAG, "检测到 ${candidates.size} 个候选框，NMS后剩余 ${detections.size} 个")
            
        } catch (e: Exception) {
            Log.e(TAG, "检测后处理失败: ${e.message}", e)
        }
        
        return detections
    }
    
    /**
     * 应用非极大值抑制（NMS）- 优化版本
     */
    private fun applyNMSOptimized(candidates: List<Detection>): List<Detection> {
        if (candidates.isEmpty()) return emptyList()
        
        // 按置信度排序
        val sortedCandidates = candidates.sortedByDescending { it.confidence }
        val selected = mutableListOf<Detection>()
        val suppressed = BooleanArray(candidates.size)
        
        for (i in sortedCandidates.indices) {
            if (suppressed[i]) continue
            
            selected.add(sortedCandidates[i])
            
            // 优化：提前退出条件
            val current = sortedCandidates[i]
            
            // 抑制与当前检测框重叠度高的其他检测框
            for (j in i + 1 until sortedCandidates.size) {
                if (suppressed[j]) continue
                
                val other = sortedCandidates[j]
                
                // 优化：快速排斥测试
                if (current.x2 < other.x1 || other.x2 < current.x1 ||
                    current.y2 < other.y1 || other.y2 < current.y1) {
                    continue
                }
                
                val iou = calculateIoU(current, other)
                if (iou > IOU_THRESHOLD) {
                    suppressed[j] = true
                }
            }
        }
        
        return selected
    }
    
    /**
     * 计算两个检测框的IoU（交并比）
     */
    private fun calculateIoU(det1: Detection, det2: Detection): Float {
        val x1 = maxOf(det1.x1, det2.x1)
        val y1 = maxOf(det1.y1, det2.y1)
        val x2 = minOf(det1.x2, det2.x2)
        val y2 = minOf(det1.y2, det2.y2)
        
        if (x2 <= x1 || y2 <= y1) return 0f
        
        val intersection = (x2 - x1) * (y2 - y1)
        val area1 = (det1.x2 - det1.x1) * (det1.y2 - det1.y1)
        val area2 = (det2.x2 - det2.x1) * (det2.y2 - det2.y1)
        val union = area1 + area2 - intersection
        
        return if (union > 0) intersection / union else 0f
    }
    
    /**
     * 后处理分割结果 - 实现真正的分割后处理（仿照Python代码）
     */
    private fun postprocessSegmentation(segOutput: FloatArray): Array<IntArray> {
        return try {
            // YOLOP分割输出格式: [batch, channels, height, width]
            // 对于可行驶区域和车道线，channels=2（背景+前景）
            val height = INPUT_SIZE
            val width = INPUT_SIZE
            
            val mask = Array(height) { IntArray(width) }
            
            // 处理分割输出 - 优化版本
            if (ENABLE_DETAILED_SEGMENTATION) {
                // 详细处理模式
                for (y in 0 until height) {
                    for (x in 0 until width) {
                        val backgroundIndex = y * width + x
                        val foregroundIndex = height * width + y * width + x
                        
                        val backgroundProb = segOutput[backgroundIndex]
                        val foregroundProb = segOutput[foregroundIndex]
                        
                        // 使用argmax逻辑：选择概率最大的类别
                        if (foregroundProb > backgroundProb) {
                            mask[y][x] = 1  // 前景
                        } else {
                            mask[y][x] = 0  // 背景
                        }
                    }
                }
            } else {
                // 快速处理模式 - 采样处理，减少计算量
                for (y in 0 until height step SEGMENTATION_SAMPLE_RATE) {
                    for (x in 0 until width step SEGMENTATION_SAMPLE_RATE) {
                        val backgroundIndex = y * width + x
                        val foregroundIndex = height * width + y * width + x
                        
                        val backgroundProb = segOutput[backgroundIndex]
                        val foregroundProb = segOutput[foregroundIndex]
                        
                        val value = if (foregroundProb > backgroundProb) 1 else 0
                        
                        // 填充周围像素
                        for (dy in 0 until SEGMENTATION_SAMPLE_RATE) {
                            for (dx in 0 until SEGMENTATION_SAMPLE_RATE) {
                                val ny = y + dy
                                val nx = x + dx
                                if (ny < height && nx < width) {
                                    mask[ny][nx] = value
                                }
                            }
                        }
                    }
                }
            }
            
            Log.d(TAG, "分割后处理完成，有效像素: ${mask.sumOf { row -> row.sum() }}")
            mask
            
        } catch (e: Exception) {
            Log.e(TAG, "分割后处理失败: ${e.message}", e)
            // 返回空掩码
            Array(INPUT_SIZE) { IntArray(INPUT_SIZE) }
        }
    }
    
    
    /**
     * 后处理车道线分割结果 - 区分实线和虚线（仿照Python代码）
     */
    private fun postprocessLaneLineSegmentation(segOutput: FloatArray): Array<IntArray> {
        return try {
            // 先进行基本的分割后处理
            val basicMask = postprocessSegmentation(segOutput)
            
            // 应用形态学处理来连接车道线（仿照Python的morphological_process + connect_lane）
            val connectedMask = applyMorphologicalProcess(basicMask)
            
            // 使用连通组件分析来区分实线和虚线（仿照Python的analyze_lane_type）
            val classifiedMask = analyzeLaneTypeWithConnectedComponents(connectedMask)
            
            Log.d(TAG, "车道线分类完成，实线像素: ${classifiedMask.sumOf { row -> row.count { it == 1 } }}")
            Log.d(TAG, "车道线分类完成，虚线像素: ${classifiedMask.sumOf { row -> row.count { it == 2 } }}")
            
            classifiedMask
            
        } catch (e: Exception) {
            Log.e(TAG, "车道线分割后处理失败: ${e.message}", e)
            // 返回空掩码
            Array(INPUT_SIZE) { IntArray(INPUT_SIZE) }
        }
    }
    
    
    /**
     * 应用形态学处理连接车道线
     */
    private fun applyMorphologicalProcess(mask: Array<IntArray>): Array<IntArray> {
        // 简化的形态学处理：使用3x3核进行膨胀和腐蚀
        val height = mask.size
        val width = mask[0].size
        val processedMask = Array(height) { IntArray(width) }
        
        // 复制原始掩码
        for (y in 0 until height) {
            for (x in 0 until width) {
                processedMask[y][x] = mask[y][x]
            }
        }
        
        // 简单的膨胀操作来连接断开的车道线
        for (y in 1 until height - 1) {
            for (x in 1 until width - 1) {
                if (mask[y][x] == 0) {
                    // 检查3x3邻域
                    var hasNeighbor = false
                    for (dy in -1..1) {
                        for (dx in -1..1) {
                            if (mask[y + dy][x + dx] == 1) {
                                hasNeighbor = true
                                break
                            }
                        }
                        if (hasNeighbor) break
                    }
                    if (hasNeighbor) {
                        processedMask[y][x] = 1
                    }
                }
            }
        }
        
        return processedMask
    }
    
    /**
     * 使用连通组件分析来区分实线和虚线（仿照Python的analyze_lane_type）
     */
    private fun analyzeLaneTypeWithConnectedComponents(mask: Array<IntArray>): Array<IntArray> {
        val height = mask.size
        val width = mask[0].size
        val solidMask = Array(height) { IntArray(width) }
        val dashedMask = Array(height) { IntArray(width) }
        
        // 简化的连通组件分析
        val components = findConnectedComponents(mask)
        
        Log.d(TAG, "🔍 发现 ${components.size} 个车道线连通组件")
        
        if (components.isEmpty()) {
            Log.d(TAG, "⚠️ 没有找到有效的车道线组件")
            return solidMask
        }
        
        // 按密度排序，通常虚线密度更低
        val sortedComponents = components.sortedBy { it.density }
        
        val totalComponents = sortedComponents.size
        
        for (idx in sortedComponents.indices) {
            val comp = sortedComponents[idx]
            
            // 计算间断性
            val discontinuityScore = calculateDiscontinuity(comp.mask, comp.width, comp.height)
            
            // 位置评分：越靠近图像中心，越可能是中央分隔虚线
            val imageCenterX = width / 2
            val centerDistance = kotlin.math.abs(comp.x + comp.width / 2 - imageCenterX).toFloat() / imageCenterX
            val positionScore = 1.0f - centerDistance
            
            // 综合判断是否为虚线（仿照Python的策略）
            var isDashed = false
            var reason = ""
            
            when {
                // 策略1: 密度极低的肯定是虚线
                comp.density < 0.06f -> {
                    isDashed = true
                    reason = "密度极低"
                }
                // 策略2: 密度较低且靠近中心
                comp.density < 0.08f && positionScore > 0.3f -> {
                    isDashed = true
                    reason = "密度低+中心位置"
                }
                // 策略3: 密度较低且间断性高
                comp.density < 0.10f && discontinuityScore > 0.2f -> {
                    isDashed = true
                    reason = "密度低+高间断性"
                }
                // 策略4: 在多组件情况下，选择密度最低的一些作为虚线
                totalComponents >= 3 && idx < totalComponents / 3 -> {
                    isDashed = true
                    reason = "相对最低密度"
                }
                // 策略5: 如果只有1-2个组件，密度最低的设为虚线
                totalComponents <= 2 && idx == 0 && comp.density < 0.12f -> {
                    isDashed = true
                    reason = "单独低密度组件"
                }
                else -> {
                    reason = "密度正常"
                }
            }
            
            // 将组件像素分配到对应的掩码
            for (y in 0 until comp.height) {
                for (x in 0 until comp.width) {
                    if (comp.mask[y][x] == 1) {
                        val globalY = comp.y + y
                        val globalX = comp.x + x
                        if (globalY < height && globalX < width) {
                            if (isDashed) {
                                dashedMask[globalY][globalX] = 1
                            } else {
                                solidMask[globalY][globalX] = 1
                            }
                        }
                    }
                }
            }
            
            Log.d(TAG, "  ${if (isDashed) "虚线" else "实线"}组件 ${comp.index}: 面积=${comp.area}, 位置=(${comp.x},${comp.y}), 尺寸=(${comp.width}x${comp.height}), 密度=${comp.density}, 原因=$reason")
        }
        
        // 合并实线和虚线掩码
        val classifiedMask = Array(height) { IntArray(width) }
        for (y in 0 until height) {
            for (x in 0 until width) {
                when {
                    solidMask[y][x] == 1 -> classifiedMask[y][x] = 1  // 实线
                    dashedMask[y][x] == 1 -> classifiedMask[y][x] = 2  // 虚线
                    else -> classifiedMask[y][x] = 0  // 背景
                }
            }
        }
        
        Log.d(TAG, "✅ 实线像素总数: ${solidMask.sumOf { row -> row.sum() }}")
        Log.d(TAG, "✅ 虚线像素总数: ${dashedMask.sumOf { row -> row.sum() }}")
        
        return classifiedMask
    }
    
    /**
     * 连通组件数据类
     */
    private data class ConnectedComponent(
        val index: Int,
        val mask: Array<IntArray>,
        val x: Int,
        val y: Int,
        val width: Int,
        val height: Int,
        val area: Int,
        val density: Float
    )
    
    /**
     * 简化的连通组件分析（仿照Python的connectedComponentsWithStats）
     */
    private fun findConnectedComponents(mask: Array<IntArray>): List<ConnectedComponent> {
        val height = mask.size
        val width = mask[0].size
        val visited = Array(height) { BooleanArray(width) }
        val components = mutableListOf<ConnectedComponent>()
        var componentIndex = 1
        
        for (y in 0 until height) {
            for (x in 0 until width) {
                if (mask[y][x] == 1 && !visited[y][x]) {
                    // 使用BFS找到连通组件
                    val component = floodFill(mask, visited, x, y, componentIndex)
                    if (component.area >= 50) { // 过滤小组件
                        components.add(component)
                        componentIndex++
                    }
                }
            }
        }
        
        return components
    }
    
    /**
     * 洪水填充算法找到连通组件
     */
    private fun floodFill(mask: Array<IntArray>, visited: Array<BooleanArray>, startX: Int, startY: Int, index: Int): ConnectedComponent {
        val height = mask.size
        val width = mask[0].size
        val queue = mutableListOf<Pair<Int, Int>>()
        val componentPixels = mutableListOf<Pair<Int, Int>>()
        
        queue.add(Pair(startX, startY))
        visited[startY][startX] = true
        
        var minX = startX
        var maxX = startX
        var minY = startY
        var maxY = startY
        
        while (queue.isNotEmpty()) {
            val (x, y) = queue.removeAt(0)
            componentPixels.add(Pair(x, y))
            
            minX = minOf(minX, x)
            maxX = maxOf(maxX, x)
            minY = minOf(minY, y)
            maxY = maxOf(maxY, y)
            
            // 检查8邻域
            for (dy in -1..1) {
                for (dx in -1..1) {
                    val nx = x + dx
                    val ny = y + dy
                    if (nx in 0 until width && ny in 0 until height && 
                        mask[ny][nx] == 1 && !visited[ny][nx]) {
                        visited[ny][nx] = true
                        queue.add(Pair(nx, ny))
                    }
                }
            }
        }
        
        val componentWidth = maxX - minX + 1
        val componentHeight = maxY - minY + 1
        val area = componentPixels.size
        val density = area.toFloat() / (componentWidth * componentHeight)
        
        // 创建组件掩码
        val componentMask = Array(componentHeight) { IntArray(componentWidth) }
        for ((px, py) in componentPixels) {
            componentMask[py - minY][px - minX] = 1
        }
        
        return ConnectedComponent(
            index = index,
            mask = componentMask,
            x = minX,
            y = minY,
            width = componentWidth,
            height = componentHeight,
            area = area,
            density = density
        )
    }
    
    /**
     * 计算组件的间断性分数（仿照Python的calculate_discontinuity）
     */
    private fun calculateDiscontinuity(componentMask: Array<IntArray>, width: Int, height: Int): Float {
        if (height < 10) return 0f
        
        // 将组件分成若干水平带，检查每带的像素密度
        val numBands = minOf(10, height / 3)
        if (numBands < 3) return 0f
        
        val bandHeight = height / numBands
        val bandDensities = mutableListOf<Float>()
        
        for (i in 0 until numBands) {
            val startY = i * bandHeight
            val endY = minOf((i + 1) * bandHeight, height)
            var bandPixels = 0
            var totalPixels = 0
            
            for (y in startY until endY) {
                for (x in 0 until width) {
                    totalPixels++
                    if (componentMask[y][x] == 1) {
                        bandPixels++
                    }
                }
            }
            
            val bandDensity = if (totalPixels > 0) bandPixels.toFloat() / totalPixels else 0f
            bandDensities.add(bandDensity)
        }
        
        // 计算密度变化的标准差
        if (bandDensities.size > 1) {
            val mean = bandDensities.average().toFloat()
            val variance = bandDensities.map { (it - mean) * (it - mean) }.average().toFloat()
            val stdDev = kotlin.math.sqrt(variance)
            val discontinuity = if (mean > 0) stdDev / mean else 0f
            return minOf(discontinuity, 1.0f)
        }
        
        return 0f
    }
    
    /**
     * 获取类别名称
     */
    private fun getClassName(classId: Int): String {
        return when (classId) {
            0 -> "car"      // 车辆
            1 -> "truck"    // 卡车
            2 -> "bus"      // 公交车
            3 -> "motorcycle" // 摩托车
            4 -> "bicycle"  // 自行车
            else -> "unknown"
        }
    }
    
    /**
     * 设置输入分辨率
     * @param resolution 新的分辨率设置
     */
    fun setInputResolution(resolution: Companion.InputResolution) {
        // 安全检查：当前模型只支持320x320
        if (resolution != Companion.InputResolution.RESOLUTION_320) {
            Log.w(TAG, "当前模型只支持320x320分辨率，自动调整为320x320")
            currentResolution = Companion.InputResolution.RESOLUTION_320
        } else {
            currentResolution = resolution
        }
        Log.i(TAG, "输入分辨率已设置为: ${currentResolution.description}")
    }
    
    /**
     * 获取当前分辨率
     */
    fun getCurrentResolution(): Companion.InputResolution = currentResolution
    
    /**
     * 释放资源
     */
    fun release() {
        try {
            onnxSession?.close()
            onnxSession = null
            onnxEnvironment = null
            isModelLoaded = false
            Log.i(TAG, "ONNX模型资源已释放")
        } catch (e: Exception) {
            Log.e(TAG, "释放资源时出错: ${e.message}", e)
        }
    }
}

/**
 * 模型输出数据类
 */
data class ModelOutputs(
    val detectionOutput: FloatArray,           // 检测输出
    val daSegOutput: FloatArray,               // 可行驶区域分割输出
    val llSegOutput: FloatArray                // 车道线分割输出
)

/**
 * YOLOP推理结果数据类
 */
data class YOLOPResult(
    val detections: List<Detection>,           // 检测结果
    val daSegMask: Array<IntArray>?,           // 可行驶区域分割掩码
    val llSegMask: Array<IntArray>?,           // 车道线分割掩码
    val inferenceTime: Long,                   // 推理耗时(ms)
    val fps: Float                             // 帧率
)

/**
 * 检测结果数据类
 */
data class Detection(
    val x1: Float,                             // 左上角x坐标
    val y1: Float,                             // 左上角y坐标
    val x2: Float,                             // 右下角x坐标
    val y2: Float,                             // 右下角y坐标
    val confidence: Float,                     // 置信度
    val classId: Int,                          // 类别ID
    val className: String                      // 类别名称
)