package org.cityman.camera.camera

import android.media.MediaCodecInfo
import android.media.MediaCodecList
import android.util.Log

/** Video resolution and frame rate the user picks; each camera offers the ones it and the encoder can do. */
enum class VideoQuality(val width: Int, val height: Int, val fps: Int, val label: String) {
    HD30(1280, 720, 30, "720p 30"),
    HD60(1280, 720, 60, "720p 60"),
    FHD30(1920, 1080, 30, "1080p 30"),
    FHD60(1920, 1080, 60, "1080p 60"),
    UHD30(3840, 2160, 30, "4K 30"),
    ;

    val area: Long get() = width.toLong() * height

    /** H.264 bit rate: 20 Mbit/s at 1080p30 as before, scaled with pixels per second (less per frame at 60). */
    val bitrate: Int
        get() = when (this) {
            HD30 -> 10_000_000
            HD60 -> 16_000_000
            FHD30 -> 20_000_000
            FHD60 -> 32_000_000
            UHD30 -> 48_000_000
        }

    /** Short form for the viewfinder badge. */
    val badge: String get() = if (this == UHD30) "4K·30" else "${height}p·$fps"

    companion object {
        val DEFAULT = FHD30

        fun fromName(name: String?): VideoQuality = entries.firstOrNull { it.name == name } ?: DEFAULT

        /**
         * The quality to record when [wanted] is not available on this camera: the same frame rate
         * at the largest size up to the wanted one, then 30 fps at the same size, then the largest
         * 30 fps one up to it, then anything.
         */
        fun pick(available: List<VideoQuality>, wanted: VideoQuality): VideoQuality? {
            if (available.isEmpty()) return null
            if (wanted in available) return wanted
            return available.filter { it.fps == wanted.fps && it.area <= wanted.area }.maxByOrNull { it.area }
                ?: available.firstOrNull { it.fps == 30 && it.area == wanted.area }
                ?: available.filter { it.fps == 30 && it.area <= wanted.area }.maxByOrNull { it.area }
                ?: available.filter { it.fps == 30 }.minByOrNull { it.area }
                ?: available.first()
        }
    }
}

/** What the phone's H.264 encoders declare (media_codecs.xml): sizes and blocks per second. */
object VideoEncoders {
    private val avc: List<MediaCodecInfo.VideoCapabilities> by lazy {
        runCatching {
            MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos
                .filter { info -> info.isEncoder && info.supportedTypes.any { it.equals("video/avc", ignoreCase = true) } }
                .mapNotNull { it.getCapabilitiesForType("video/avc").videoCapabilities }
        }.onFailure { Log.w("ArcoCamera", "encoder list", it) }.getOrDefault(emptyList())
    }

    /** For the log: each encoder's size and frame-rate limits. */
    fun describe(): String = avc.joinToString { c ->
        "${c.supportedWidths.upper}x${c.supportedHeights.upper}, ${c.supportedFrameRates}, 4K: ${runCatching { c.getSupportedFrameRatesFor(3840, 2160) }.getOrNull()}"
    }.ifEmpty { "none listed" }

    /** True when an H.264 encoder takes this size and rate; unknown (no list) counts as yes. */
    fun supports(width: Int, height: Int, fps: Int): Boolean =
        avc.isEmpty() || avc.any { runCatching { it.areSizeAndRateSupported(width, height, fps.toDouble()) }.getOrDefault(false) }
}
