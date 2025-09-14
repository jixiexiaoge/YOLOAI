package com.example.yoloai.webrtc

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withTimeout
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import org.webrtc.*
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.roundToInt

/**
 * WebRtcViewer: 连接 Comma3 dashy 的 /stream 信令接口，接收 road 摄像头视频并渲染。
 * 仅接收视频（recvonly），不申请摄像头/麦克风权限。
 */
class WebRtcViewer(
    private val context: Context,
    private val surfaceViewRenderer: SurfaceViewRenderer,
    private val deviceIp: String,
) {
    companion object {
        private const val TAG = "WebRtcViewer"
        private const val SIGNAL_URL_FMT = "http://%s:5001/stream"
    }

    private val eglBase: EglBase = EglBase.create()
    private var peerConnectionFactory: PeerConnectionFactory? = null
    private var peerConnection: PeerConnection? = null
    private var videoSink: VideoSink? = null
    private var aspectRatioSet: Boolean = false
    private var scope: CoroutineScope? = null

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    fun start() {
        stop()
        Log.i(TAG, "start() deviceIp=$deviceIp")
        scope = CoroutineScope(Dispatchers.Main + Job())
        initPeerConnectionFactory()
        initRenderer()
        scope?.launch {
            try {
                val pc = createPeerConnection() ?: return@launch
                Log.i(TAG, "PeerConnection created")
                
                // 使用协程化的SDP创建方法
                val offer = withContext(Dispatchers.Main) {
                    createOfferAsync(pc)
                }
                
                // 发送信令并获取答案
                val answer = withContext(Dispatchers.IO) {
                    signalAndGetAnswer(offer.description)
                }
                
                // 设置远程描述
                withContext(Dispatchers.Main) {
                    setRemoteDescriptionAsync(pc, answer)
                }
                
                Log.i(TAG, "WebRTC connection setup completed - waiting for video stream...")
                
                // 等待连接建立
                delay(2000) // 给连接一些时间建立
                
                // 检查连接状态
                val connectionState = pc.connectionState()
                val iceConnectionState = pc.iceConnectionState()
                Log.i(TAG, "Connection state: $connectionState, ICE state: $iceConnectionState")
            } catch (e: Exception) {
                Log.e(TAG, "start error", e)
            }
        }
    }

    fun stop() {
        try {
            scope?.let { s ->
                s.coroutineContext[Job]?.cancel()
            }
        } catch (_: Throwable) {}
        scope = null
        try {
            peerConnection?.close()
        } catch (_: Throwable) {}
        peerConnection = null
        try {
            surfaceViewRenderer.release()
        } catch (_: Throwable) {}
        try {
            peerConnectionFactory?.dispose()
        } catch (_: Throwable) {}
        peerConnectionFactory = null
    }

    private fun initPeerConnectionFactory() {
        if (peerConnectionFactory != null) return
        val initializationOptions = PeerConnectionFactory.InitializationOptions.builder(context)
            .setEnableInternalTracer(false)
            .createInitializationOptions()
        PeerConnectionFactory.initialize(initializationOptions)

        val encoderFactory = DefaultVideoEncoderFactory(eglBase.eglBaseContext, true, true)
        val decoderFactory = DefaultVideoDecoderFactory(eglBase.eglBaseContext)
        peerConnectionFactory = PeerConnectionFactory.builder()
            .setVideoEncoderFactory(encoderFactory)
            .setVideoDecoderFactory(decoderFactory)
            .createPeerConnectionFactory()
    }

    private fun initRenderer() {
        try {
            // 初始化SurfaceViewRenderer
        surfaceViewRenderer.init(eglBase.eglBaseContext, null)
        surfaceViewRenderer.setEnableHardwareScaler(true)
        surfaceViewRenderer.setMirror(false)
        surfaceViewRenderer.setScalingType(RendererCommon.ScalingType.SCALE_ASPECT_FIT)
            
            // 重置首帧比例设置标记
            aspectRatioSet = false

            // 创建视频接收器：首帧动态调整宽高比，避免裁切
            videoSink = VideoSink { frame ->
                if (!aspectRatioSet) {
                    val w = frame.rotatedWidth
                    val h = frame.rotatedHeight
                    if (w > 0 && h > 0) {
                        val ratio = w.toFloat() / h.toFloat()
                        surfaceViewRenderer.post {
                            // 再次确保使用等比适配
                            surfaceViewRenderer.setScalingType(RendererCommon.ScalingType.SCALE_ASPECT_FIT)
                            // 按照首帧宽高比调整视图高度，避免裁切
                            val lp = surfaceViewRenderer.layoutParams
                            val currentWidth = surfaceViewRenderer.width.takeIf { it > 0 } ?: lp.width
                            if (currentWidth > 0 && lp.height != (currentWidth / ratio).roundToInt()) {
                                lp.height = (currentWidth / ratio).roundToInt()
                                surfaceViewRenderer.layoutParams = lp
                                surfaceViewRenderer.requestLayout()
                            }
                        }
                        aspectRatioSet = true
                        Log.i(TAG, "Set layout aspect ratio to ${'$'}ratio for first frame ${'$'}w x ${'$'}h")
                    }
                }
                // Log.d(TAG, "Received video frame: ${'$'}{frame.rotatedWidth}x${'$'}{frame.rotatedHeight}")
            }
            
            Log.i(TAG, "Renderer initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize renderer", e)
            throw e
        }
    }

    /**
     * 协程化的SDP Offer创建方法
     * 使用suspendCancellableCoroutine来正确处理异步回调
     */
    private suspend fun createOfferAsync(pc: PeerConnection): SessionDescription = 
        withTimeout(15000) { // 15秒超时
            suspendCancellableCoroutine { continuation ->
                Log.i(TAG, "Creating offer with constraints...")
                val constraints = MediaConstraints().apply {
                    // 明确指定只接收视频，不接收音频 - 与webrtcd.py兼容
                    mandatory.add(MediaConstraints.KeyValuePair("OfferToReceiveAudio", "false"))
                    mandatory.add(MediaConstraints.KeyValuePair("OfferToReceiveVideo", "true"))
                    // 添加H.264编解码器支持
                    mandatory.add(MediaConstraints.KeyValuePair("googCpuOveruseDetection", "true"))
                    mandatory.add(MediaConstraints.KeyValuePair("googEchoCancellation", "false"))
                    mandatory.add(MediaConstraints.KeyValuePair("googAutoGainControl", "false"))
                    mandatory.add(MediaConstraints.KeyValuePair("googNoiseSuppression", "false"))
                }
                
                val sdpObserver = object : SdpObserver {
                    override fun onCreateSuccess(sdp: SessionDescription?) {
                        if (sdp != null) {
                            Log.i(TAG, "createOffer success: type=${sdp.type}, len=${sdp.description?.length}")
                            // Log.d(TAG, "Offer SDP preview: ${sdp.description?.take(300)}...")
                            
                            // 优化SDP，确保H.264编解码器优先
                            val optimizedSdp = optimizeSdpForH264(sdp.description ?: "")
                            val optimizedOffer = SessionDescription(sdp.type, optimizedSdp)
                            
                            // 设置本地描述
                            pc.setLocalDescription(object : SdpObserver {
                                override fun onCreateSuccess(sdp: SessionDescription?) {}
                                override fun onSetSuccess() {
                                    Log.i(TAG, "setLocalDescription success")
                                    if (continuation.isActive) {
                                        continuation.resume(optimizedOffer)
                                    }
                                }
                                override fun onCreateFailure(error: String?) {
                                    Log.e(TAG, "setLocalDescription createFailure: $error")
                                    if (continuation.isActive) {
                                        continuation.resumeWithException(RuntimeException("setLocalDescription failed: $error"))
                                    }
                                }
                                override fun onSetFailure(error: String?) {
                                    Log.e(TAG, "setLocalDescription setFailure: $error")
                                    if (continuation.isActive) {
                                        continuation.resumeWithException(RuntimeException("setLocalDescription failed: $error"))
                                    }
                                }
                            }, optimizedOffer)
                        } else {
                            Log.e(TAG, "createOffer returned null SDP")
                            if (continuation.isActive) {
                                continuation.resumeWithException(RuntimeException("createOffer returned null SDP"))
                            }
                        }
                    }
                    
                    override fun onSetSuccess() {
                        // Log.d(TAG, "createOffer onSetSuccess")
                    }
                    
                    override fun onCreateFailure(error: String?) {
                        Log.e(TAG, "createOffer failure: $error")
                        if (continuation.isActive) {
                            continuation.resumeWithException(RuntimeException("createOffer failed: $error"))
                        }
                    }
                    
                    override fun onSetFailure(error: String?) {
                        Log.e(TAG, "createOffer setFailure: $error")
                        if (continuation.isActive) {
                            continuation.resumeWithException(RuntimeException("createOffer set failed: $error"))
                        }
                    }
                }
                
                // Log.i(TAG, "Calling pc.createOffer...")
                pc.createOffer(sdpObserver, constraints)
                
                // 设置取消回调
                continuation.invokeOnCancellation {
                    Log.w(TAG, "createOfferAsync cancelled")
                }
            }
        }

    /**
     * 协程化的远程描述设置方法
     */
    private suspend fun setRemoteDescriptionAsync(pc: PeerConnection, answerSdp: String): Unit =
        withTimeout(10000) { // 10秒超时
            suspendCancellableCoroutine { continuation ->
                val answer = SessionDescription(SessionDescription.Type.ANSWER, answerSdp)
                Log.i(TAG, "Setting remote description, sdpLen=${answerSdp.length}")
                
                pc.setRemoteDescription(object : SdpObserver {
                    override fun onCreateSuccess(sdp: SessionDescription?) {}
                    override fun onSetSuccess() {
                        Log.i(TAG, "setRemoteDescription success")
                        if (continuation.isActive) {
                            continuation.resume(Unit)
                        }
                    }
                    override fun onCreateFailure(error: String?) {
                        Log.e(TAG, "setRemoteDescription createFailure: $error")
                        if (continuation.isActive) {
                            continuation.resumeWithException(RuntimeException("setRemoteDescription failed: $error"))
                        }
                    }
                    override fun onSetFailure(error: String?) {
                        Log.e(TAG, "setRemoteDescription setFailure: $error")
                        if (continuation.isActive) {
                            continuation.resumeWithException(RuntimeException("setRemoteDescription failed: $error"))
                        }
                    }
                }, answer)
                
                // 设置取消回调
                continuation.invokeOnCancellation {
                    Log.w(TAG, "setRemoteDescriptionAsync cancelled")
                }
            }
    }

    private fun createPeerConnection(): PeerConnection? {
        // 配置ICE服务器和连接参数 - 针对Comma3设备优化
        val rtcConfig = PeerConnection.RTCConfiguration(listOf(
            PeerConnection.IceServer.builder("stun:stun.l.google.com:19302").createIceServer(),
            PeerConnection.IceServer.builder("stun:stun1.l.google.com:19302").createIceServer()
        )).apply {
            sdpSemantics = PeerConnection.SdpSemantics.UNIFIED_PLAN
            continualGatheringPolicy = PeerConnection.ContinualGatheringPolicy.GATHER_CONTINUALLY
            // 针对局域网连接优化
            iceConnectionReceivingTimeout = 10000
            iceBackupCandidatePairPingInterval = 5000
            // 启用ICE重启和TURN
            iceCandidatePoolSize = 10
        }
        
        val pc = peerConnectionFactory?.createPeerConnection(rtcConfig, object : PeerConnection.Observer {
            override fun onSignalingChange(newState: PeerConnection.SignalingState) {
                Log.i(TAG, "onSignalingChange=$newState")
                when (newState) {
                    PeerConnection.SignalingState.STABLE -> {
                        Log.i(TAG, "Signaling STABLE - connection established")
                    }
                    PeerConnection.SignalingState.HAVE_LOCAL_OFFER -> {
                        Log.i(TAG, "Signaling HAVE_LOCAL_OFFER - waiting for remote answer")
                    }
                    PeerConnection.SignalingState.HAVE_REMOTE_OFFER -> {
                        Log.i(TAG, "Signaling HAVE_REMOTE_OFFER")
                    }
                    PeerConnection.SignalingState.HAVE_LOCAL_PRANSWER -> {
                        Log.i(TAG, "Signaling HAVE_LOCAL_PRANSWER")
                    }
                    PeerConnection.SignalingState.HAVE_REMOTE_PRANSWER -> {
                        Log.i(TAG, "Signaling HAVE_REMOTE_PRANSWER")
                    }
                    PeerConnection.SignalingState.CLOSED -> {
                        Log.w(TAG, "Signaling CLOSED")
                    }
                }
            }
            
            override fun onIceConnectionChange(newState: PeerConnection.IceConnectionState) {
                Log.i(TAG, "onIceConnectionChange=$newState")
                when (newState) {
                    PeerConnection.IceConnectionState.CONNECTED -> {
                        Log.i(TAG, "ICE CONNECTED! Video should start flowing")
                    }
                    PeerConnection.IceConnectionState.COMPLETED -> {
                        Log.i(TAG, "ICE COMPLETED! Connection fully established")
                    }
                    PeerConnection.IceConnectionState.FAILED -> {
                        Log.e(TAG, "ICE CONNECTION FAILED! Check network connectivity")
                        // 可以在这里添加重连逻辑
                    }
                    PeerConnection.IceConnectionState.DISCONNECTED -> {
                        Log.w(TAG, "ICE DISCONNECTED - attempting to reconnect")
                    }
                    PeerConnection.IceConnectionState.CLOSED -> {
                        Log.w(TAG, "ICE CLOSED")
                    }
                    PeerConnection.IceConnectionState.CHECKING -> {
                        Log.i(TAG, "ICE CHECKING - establishing connection")
                    }
                    PeerConnection.IceConnectionState.NEW -> {
                        Log.i(TAG, "ICE NEW - starting connection process")
                    }
                    else -> {
                        // Log.d(TAG, "ICE state: $newState")
                    }
                }
            }
            
            override fun onIceConnectionReceivingChange(receiving: Boolean) {
                Log.i(TAG, "onIceConnectionReceivingChange=$receiving")
            }
            
            override fun onIceGatheringChange(newState: PeerConnection.IceGatheringState) {
                Log.i(TAG, "onIceGatheringChange=$newState")
            }
            
            override fun onIceCandidate(candidate: IceCandidate) {
                // Log.d(TAG, "onIceCandidate: ${candidate.sdp}")
            }
            
            override fun onIceCandidatesRemoved(candidates: Array<out IceCandidate>) {
                Log.i(TAG, "onIceCandidatesRemoved count=${candidates.size}")
            }
            
            override fun onAddStream(stream: MediaStream) {
                Log.i(TAG, "onAddStream id=${stream.id} videoTracks=${stream.videoTracks.size} audioTracks=${stream.audioTracks.size}")
                // 处理传统addStream回调（兼容性）
                for (videoTrack in stream.videoTracks) {
                    Log.i(TAG, "Adding VideoTrack from stream: ${videoTrack.id()}")
                    videoTrack.setEnabled(true)
                    // 直接添加到SurfaceViewRenderer
                    videoTrack.addSink(surfaceViewRenderer)
                    // 同时添加到VideoSink用于日志记录
                    videoTrack.addSink(videoSink)
                    Log.i(TAG, "VideoTrack from stream added to renderer")
                }
            }
            
            override fun onRemoveStream(stream: MediaStream) {
                Log.i(TAG, "onRemoveStream id=${stream.id}")
            }
            
            override fun onDataChannel(dc: DataChannel) { 
                Log.i(TAG, "onDataChannel opened label=${dc.label()} state=${dc.state()}") 
            }
            
            override fun onRenegotiationNeeded() { 
                Log.i(TAG, "onRenegotiationNeeded") 
            }
            
            override fun onAddTrack(receiver: RtpReceiver, streams: Array<out MediaStream>) {
                val track = receiver.track()
                Log.i(TAG, "onAddTrack kind=${track?.kind()} id=${track?.id()} enabled=${track?.enabled()} streams=${streams.size}")
                
                if (track is VideoTrack) {
                    Log.i(TAG, "Adding VideoTrack to sink - track enabled: ${track.enabled()}")
                    // 确保视频轨道启用
                    track.setEnabled(true)
                    
                    // 直接添加到SurfaceViewRenderer，而不是通过VideoSink
                    track.addSink(surfaceViewRenderer)
                    Log.i(TAG, "VideoTrack added directly to SurfaceViewRenderer")
                    
                    // 同时添加到我们的VideoSink用于日志记录
                    track.addSink(videoSink)
                    Log.i(TAG, "VideoTrack added to sink successfully")
                } else {
                    Log.w(TAG, "Track is not VideoTrack: ${track?.javaClass}")
                }
            }
        }) ?: return null

        // 添加视频接收器 - 明确指定只接收视频
        val transceiverInit = RtpTransceiver.RtpTransceiverInit(RtpTransceiver.RtpTransceiverDirection.RECV_ONLY)
        val transceiver = pc.addTransceiver(MediaStreamTrack.MediaType.MEDIA_TYPE_VIDEO, transceiverInit)
        Log.i(TAG, "Added video transceiver: ${transceiver.direction}")
        
        peerConnection = pc
        return pc
    }

    private fun signalAndGetAnswer(offerSdp: String): String {
        val url = SIGNAL_URL_FMT.format(deviceIp)
        Log.i(TAG, "signalAndGetAnswer POST to $url")
        
        try {
            // 发送webrtcd.py期望的格式 - 与StreamRequestBody完全匹配
            val json = JSONObject().apply {
                put("sdp", offerSdp)
                // 使用JSONArray而不是listOf，确保正确的JSON格式
                val camerasArray = org.json.JSONArray()
                camerasArray.put("road")
                put("cameras", camerasArray)
                
                val bridgeServicesInArray = org.json.JSONArray()
                put("bridge_services_in", bridgeServicesInArray)
                
                val bridgeServicesOutArray = org.json.JSONArray()
                bridgeServicesOutArray.put("modelV2")
                bridgeServicesOutArray.put("liveCalibration")
                bridgeServicesOutArray.put("longitudinalPlan")
                bridgeServicesOutArray.put("radarState")
                bridgeServicesOutArray.put("selfdriveState")
                bridgeServicesOutArray.put("deviceState")
                bridgeServicesOutArray.put("carState")
                bridgeServicesOutArray.put("controlsState")
                put("bridge_services_out", bridgeServicesOutArray)
            }
            
            Log.i(TAG, "signalAndGetAnswer sending request, sdpLen=${offerSdp.length}")
            // Log.d(TAG, "Request JSON: ${json.toString()}")
            // Log.d(TAG, "Offer SDP preview: ${offerSdp.take(200)}...")
            
            val body = json.toString().toRequestBody("application/json; charset=utf-8".toMediaType())
            val req = Request.Builder()
                .url(url)
                .post(body)
                .addHeader("Content-Type", "application/json")
                .addHeader("Accept", "application/json")
                .build()
                
            httpClient.newCall(req).execute().use { resp ->
                val bodyStr = resp.body?.string()
                Log.i(TAG, "HTTP Response: code=${resp.code}, headers=${resp.headers}")
                
                if (!resp.isSuccessful) {
                    Log.e(TAG, "signalAndGetAnswer FAIL code=${resp.code} body=$bodyStr")
                    Log.e(TAG, "Request URL: $url")
                    Log.e(TAG, "Request JSON: ${json.toString()}")
                    throw IllegalStateException("Signal HTTP ${resp.code}: $bodyStr")
                }
                
                val text = bodyStr ?: throw IllegalStateException("Empty response body")
                Log.i(TAG, "signalAndGetAnswer SUCCESS HTTP ${resp.code}, bodyLen=${text.length}")
                // Log.d(TAG, "Response body: $text")
                
                // 解析JSON响应，提取SDP
                val responseJson = JSONObject(text)
                val answerSdp = responseJson.getString("sdp")
                val answerType = responseJson.getString("type")
                Log.i(TAG, "signalAndGetAnswer parsed answer type=$answerType, sdpLen=${answerSdp.length}")
                // Log.d(TAG, "Answer SDP preview: ${answerSdp.take(200)}...")
                
                return answerSdp
            }
        } catch (e: Exception) {
            Log.e(TAG, "signalAndGetAnswer error", e)
            throw e
        }
    }
    
    private fun optimizeSdpForH264(sdp: String): String {
        // 优化SDP以确保与webrtcd.py的teleoprtc库兼容
        var optimizedSdp = sdp
        
        try {
            val lines = optimizedSdp.split("\n").toMutableList()
            val optimizedLines = mutableListOf<String>()
            var inVideoSection = false
            
            for (line in lines) {
                when {
                    // 视频媒体行
                    line.startsWith("m=video") -> {
                        optimizedLines.add(line)
                        inVideoSection = true
                        Log.i(TAG, "Found video media line: $line")
                    }
                    // 音频媒体行 - 设置为inactive
                    line.startsWith("m=audio") -> {
                        optimizedLines.add("m=audio 0 RTP/SAVPF") // 设置为inactive
                        inVideoSection = false
                        Log.i(TAG, "Set audio to inactive")
                    }
                    // 修改方向为recvonly
                    line.startsWith("a=sendrecv") -> {
                        optimizedLines.add("a=recvonly") // 只接收视频
                        Log.i(TAG, "Changed direction to recvonly")
                    }
                    // 在视频部分添加H.264编解码器配置
                    inVideoSection && line.startsWith("a=rtpmap:") -> {
                        optimizedLines.add(line)
                        // 如果还没有H.264，添加它
        if (!optimizedSdp.contains("H264")) {
                            Log.i(TAG, "Adding H.264 codec configuration")
                            optimizedLines.add("a=rtpmap:96 H264/90000")
                            optimizedLines.add("a=fmtp:96 profile-level-id=42e01f;packetization-mode=1")
                        }
                    }
                    // 其他行保持不变
                    else -> {
                        optimizedLines.add(line)
                    }
                }
            }
            
            optimizedSdp = optimizedLines.joinToString("\n")
            // Log.d(TAG, "SDP optimized for webrtcd.py compatibility")
            // Log.d(TAG, "Optimized SDP preview: ${optimizedSdp.take(500)}...")
            
        } catch (e: Exception) {
            Log.w(TAG, "SDP optimization failed, using original SDP", e)
        }
        
        return optimizedSdp
    }
}

/**
 * 简单的SDP观察者基类，用于处理SDP操作的回调
 */
private open class SimpleSdpObserver : SdpObserver {
    override fun onCreateSuccess(sdp: SessionDescription?) {}
    override fun onSetSuccess() {}
    override fun onCreateFailure(error: String?) {}
    override fun onSetFailure(error: String?) {}
}
