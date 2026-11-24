package org.cityman.camera.camera

import android.graphics.ImageFormat
import android.graphics.Rect
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.media.MediaRecorder
import android.util.Range
import android.util.Size
import android.view.SurfaceHolder
import kotlin.math.abs

enum class PhotoFormat { FULL, RATIO_4_3, RATIO_16_9 }

enum class MultiFrameBlocker { NO_RAW, NO_MANUAL, NO_GPU }

/** What one camera can do, read once from its characteristics. */
class CameraCaps(val id: String, val chars: CameraCharacteristics) {
    val front = chars[CameraCharacteristics.LENS_FACING] == CameraMetadata.LENS_FACING_FRONT
    val sensorOrientation: Int = chars[CameraCharacteristics.SENSOR_ORIENTATION] ?: 90
    val activeArray: Rect = chars[CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE] ?: Rect(0, 0, 1, 1)
    val isoRange: Range<Int>? = chars[CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE]
    val exposureRange: Range<Long>? = chars[CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE]
    val evRange: Range<Int> = chars[CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE] ?: Range(0, 0)
    val evStep: Float = chars[CameraCharacteristics.CONTROL_AE_COMPENSATION_STEP]?.toFloat() ?: 0f
    val minFocusDiopters: Float = chars[CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE] ?: 0f
    val afModes: Set<Int> = chars[CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES]?.toSet() ?: emptySet()
    private val capabilities = chars[CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES]?.toSet() ?: emptySet()
    val manualSensor = CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR in capabilities
    val manualPostProcessing = CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_POST_PROCESSING in capabilities
    val maxZoom: Float = chars[CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM] ?: 1f
    val maxAfRegions: Int = chars[CameraCharacteristics.CONTROL_MAX_REGIONS_AF] ?: 0
    val maxAeRegions: Int = chars[CameraCharacteristics.CONTROL_MAX_REGIONS_AE] ?: 0
    val fpsRanges: List<Range<Int>> = chars[CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES]?.toList() ?: emptyList()
    val color: ColorCalibration? = ColorCalibration.from(chars)

    private val map = chars[CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP]!!
    private val jpegSizes: List<Size> = map.getOutputSizes(ImageFormat.JPEG)?.toList() ?: emptyList()
    private val previewSizes: List<Size> = map.getOutputSizes(SurfaceHolder::class.java)?.toList() ?: emptyList()
    private val videoSizes: List<Size> = map.getOutputSizes(MediaRecorder::class.java)?.toList() ?: emptyList()
    private val yuvSizes: List<Size> = map.getOutputSizes(ImageFormat.YUV_420_888)?.toList() ?: emptyList()

    /** Small YUV size with the viewfinder's shape, for frame analysis. */
    fun analysisSize(preview: Size): Size? {
        val ratio = preview.width.toFloat() / preview.height
        return yuvSizes.filter { it.width in 320..640 }
            .minWithOrNull(compareBy<Size> { (abs(it.width.toFloat() / it.height - ratio) * 50).toInt() }.thenByDescending { it.area() })
    }

    val rawSize: Size? = if (CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_RAW in capabilities) {
        map.getOutputSizes(ImageFormat.RAW_SENSOR)?.maxByOrNull { it.area() }
    } else null

    /** LEGACY, LIMITED, FULL, LEVEL_3 or EXTERNAL, as the HAL reports it. */
    val hardwareLevel: Int = chars[CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL]
        ?: CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY
    val hardwareLevelName: String = when (hardwareLevel) {
        CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY -> "LEGACY"
        CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED -> "LIMITED"
        CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_FULL -> "FULL"
        CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_3 -> "LEVEL_3"
        CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_EXTERNAL -> "EXTERNAL"
        else -> "level $hardwareLevel"
    }
    private val requestKeys = chars.availableCaptureRequestKeys.toSet()

    /** Manual focus needs a movable lens and a HAL that honours the focus distance. */
    val hasFocus: Boolean
        get() = minFocusDiopters > 0f && CameraMetadata.CONTROL_AF_MODE_OFF in afModes &&
            CaptureRequest.LENS_FOCUS_DISTANCE in requestKeys

    /**
     * Why multi-frame RAW is off on this camera, or null when it can run: it needs RAW frames,
     * manual exposure for the burst and compute shaders on the GPU.
     */
    val multiFrameBlocker: MultiFrameBlocker?
        get() = when {
            rawSize == null -> MultiFrameBlocker.NO_RAW
            !manualSensor -> MultiFrameBlocker.NO_MANUAL
            !DeviceSupport.computeShaders -> MultiFrameBlocker.NO_GPU
            else -> null
        }
    val multiFrameReady: Boolean get() = multiFrameBlocker == null

    /** One line for the log and the settings page. */
    fun summary(): String = buildString {
        append(hardwareLevelName)
        if (rawSize != null) append(" · RAW")
        if (manualSensor) append(" · manual")
    }
    val hasAutoFocus: Boolean get() = CameraMetadata.CONTROL_AF_MODE_AUTO in afModes

    fun jpegSize(format: PhotoFormat): Size {
        val largest = jpegSizes.maxBy { it.area() }
        return when (format) {
            PhotoFormat.FULL -> largest
            PhotoFormat.RATIO_4_3 -> jpegSizes.filter { it.ratioNear(4f / 3f) }.maxByOrNull { it.area() } ?: largest
            PhotoFormat.RATIO_16_9 -> jpegSizes.filter { it.ratioNear(16f / 9f) }.maxByOrNull { it.area() } ?: largest
        }
    }

    /** Largest viewfinder size up to 1080p whose shape is closest to [target]. */
    fun previewSize(target: Size): Size {
        val ratio = target.width.toFloat() / target.height
        val candidates = previewSizes.filter { it.width <= 1920 && it.height <= 1080 }.ifEmpty { previewSizes }
        return candidates.minWith(
            compareBy<Size> { (abs(it.width.toFloat() / it.height - ratio) * 50).toInt() }
                .thenByDescending { it.area() },
        )
    }

    fun videoSize(): Size {
        val wanted = listOf(Size(1920, 1080), Size(1280, 720))
        return wanted.firstOrNull { it in videoSizes && it in previewSizes }
            ?: videoSizes.filter { it.ratioNear(16f / 9f) && it.width <= 1920 }.maxByOrNull { it.area() }
            ?: videoSizes.first()
    }

    val legacy: Boolean get() = hardwareLevel == CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY

    private fun listed(list: List<Size>, w: Int, h: Int) = list.any { it.width == w && it.height == h }

    /**
     * AE range for video at [fps]: the usual one at 30 fps; above that a fixed range, or one that
     * ends there. Only listed ranges: Android's LEGACY layer drops any other (HAL1
     * `preview-fps-range-values`), and HAL3 may refuse it.
     */
    private fun videoRange(fps: Int): Range<Int>? =
        if (fps <= 30) fpsRange(video = true)
        else fpsRanges.firstOrNull { it.lower == fps && it.upper == fps }
            ?: fpsRanges.filter { it.upper == fps }.maxByOrNull { it.lower }

    private fun frameDurationOk(q: VideoQuality): Boolean {
        val d = runCatching { map.getOutputMinFrameDuration(MediaRecorder::class.java, Size(q.width, q.height)) }.getOrDefault(0L)
        return d <= 0L || d <= 1_000_000_000L / q.fps * 21 / 20
    }

    /**
     * Video qualities this camera can record: the size is offered for the recorder and for the
     * YUV path, the frame rate is in the AE ranges and fits the stream's minimum frame duration,
     * and an H.264 encoder takes it. 720p30 and 1080p30 skip the encoder check, as before.
     */
    val videoQualities: List<VideoQuality> by lazy { VideoQuality.entries.filter { videoQualityProblem(it) == null } }

    /** Why this camera does not offer [q], or null when it does (logged with the session). */
    fun videoQualityProblem(q: VideoQuality): String? = when {
        !listed(videoSizes, q.width, q.height) -> "no recorder size"
        !listed(yuvSizes, q.width, q.height) && !listed(previewSizes, q.width, q.height) -> "no YUV/preview size"
        q.fps > 30 && videoRange(q.fps) == null -> "no ${q.fps} fps AE range"
        !frameDurationOk(q) -> "min frame duration ${runCatching { map.getOutputMinFrameDuration(MediaRecorder::class.java, Size(q.width, q.height)) }.getOrDefault(0L) / 1000} µs"
        q != VideoQuality.HD30 && q != VideoQuality.FHD30 && !VideoEncoders.supports(q.width, q.height, q.fps) -> "H.264 encoder: ${VideoEncoders.describe()}"
        else -> null
    }

    /** What to record for the user's [wanted] quality on this camera. */
    fun videoConfig(wanted: VideoQuality): VideoConfig {
        val q = VideoQuality.pick(videoQualities, wanted)
            ?: return videoSize().let { VideoConfig(it, 30, if (it.width >= 1920) 20_000_000 else 10_000_000, fpsRange(video = true), null) }
        return VideoConfig(Size(q.width, q.height), q.fps, q.bitrate, videoRange(q.fps), q)
    }

    /**
     * LEGACY cameras (HAL1 through Android's compatibility layer): that layer picks the HAL's
     * preview size from the shape of the session's JPEG stream, the full 4:3 photo when there is
     * none, so a 16:9 video ran on the largest preview mode (4096x2160 on the Lumia 930, 25 fps).
     * The smallest JPEG with the video's shape makes it pick the matching 16:9 mode instead.
     */
    fun jpegSizeLike(video: Size): Size? {
        val r = video.width.toFloat() / video.height
        return jpegSizes.filter { abs(it.width.toFloat() / it.height - r) < 0.04f }.minByOrNull { it.area() }
    }

    /** Widest frame-rate range for photos (longer exposures in low light), fixed and fastest for video. */
    fun fpsRange(video: Boolean): Range<Int>? {
        if (fpsRanges.isEmpty()) return null
        return if (video) {
            // Never a 60 fps range for 30 fps video: HAL1's 31.5 fps rounds up to 32 here.
            fpsRanges.filter { it.lower == it.upper && it.upper <= 30 }.maxByOrNull { it.upper }
                ?: fpsRanges.filter { it.upper <= 32 }.maxByOrNull { it.upper }
                ?: fpsRanges.minBy { it.upper }
        } else {
            fpsRanges.maxWith(compareBy<Range<Int>> { it.upper - it.lower }.thenBy { it.upper })
        }
    }

    fun minFrameDuration(format: Int, size: Size): Long = map.getOutputMinFrameDuration(format, size)

    companion object {
        fun Size.area() = width.toLong() * height
        fun Size.ratioNear(r: Float) = abs(width.toFloat() / height - r) < 0.02f
    }
}

/** Video stream and recorder settings for one recording; [quality] is null on cameras that offer none of the listed sizes. */
class VideoConfig(val size: Size, val fps: Int, val bitrate: Int, val range: Range<Int>?, val quality: VideoQuality?)

/**
 * Manual overrides; null means automatic. Exposure compensation is in Camera2 steps and only
 * applies while ISO or shutter (or both) are automatic.
 */
data class ManualControls(
    val wbKelvin: Float? = null,
    val focusDiopters: Float? = null,
    val iso: Int? = null,
    val exposureNs: Long? = null,
    val evSteps: Int = 0,
)

/** Values reported by the camera for the latest viewfinder frame. */
data class LiveValues(
    val iso: Int = 0,
    val exposureNs: Long = 0,
    val focusDiopters: Float = 0f,
    val kelvin: Float? = null,
    val afState: Int = 0,
)
