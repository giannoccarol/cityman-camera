package org.cityman.camera.camera

import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt

/** Exposure and length of a multi-frame burst. */
data class BurstPlan(
    val iso: Int,
    val exposureNs: Long,
    /** Digital gain to apply after the merge. */
    val gain: Float,
    val frames: Int,
    val underEv: Float = BurstPlanner.UNDER_EV.toFloat(),
    /** Extra highlight frames (bracketing) at [shortExposureNs], appended after [frames]. */
    val shortFrames: Int = 0,
    val shortExposureNs: Long = 0,
)

/** What the planner needs to know, in plain types (no Android classes, so it runs in unit tests). */
data class PlanInput(
    /** Manual ISO range the HAL accepts (342 at most on this rear camera). */
    val isoMin: Int,
    val isoMax: Int,
    val exposureMinNs: Long,
    val exposureMaxNs: Long,
    /** What auto exposure is using now, and the exposure compensation in EV. */
    val autoIso: Int,
    val autoExposureNs: Long,
    val evCompensation: Double = 0.0,
    /** Manual ISO / shutter already resolved by the controller, or null for automatic. */
    val manual: Pair<Int, Long>? = null,
    /** Share of blown pixels in the viewfinder, negative when unknown. */
    val clipped: Float = -1f,
    /** Hand shake (rad/s) and whether the gyroscope can be trusted (it has moved at least once). */
    val gyroRate: Double = 0.0,
    val gyroTrusted: Boolean = false,
    /** Phone resting (tripod, table). */
    val still: Boolean = false,
    val rawWidth: Int = 5344,
    val front: Boolean = false,
    /** Thermal headroom low: shorter bursts. */
    val hot: Boolean = false,
    /** Frames chosen in the settings, 0 = automatic. */
    val chosenFrames: Int = 0,
)

/**
 * Burst exposure. Automatic: under what auto exposure measured (0.4–1.6 EV, more when the
 * viewfinder shows blown highlights) to keep highlights; short frames against hand shake; ISO up
 * to the HAL's manual maximum, then longer frames up to 1/10 s handheld (1/2 s on a tripod), then
 * digital gain after the merge, where the noise is lower.
 */
object BurstPlanner {
    const val MAX_FRAMES = 12
    const val UNDER_EV = 0.7
    const val SHUTTER_NS = 33_333_333L
    const val LOW_LIGHT_SHUTTER_NS = 100_000_000L
    const val NIGHT_SHUTTER_NS = 500_000_000L
    const val DIGITAL_GAIN_FIRST = 2.0

    fun plan(p: PlanInput): BurstPlan {
        // Focal length in RAW pixels (both lenses are about 70° wide) and the blur accepted per
        // frame: 2 px, twice that behind the rear lens' optical stabiliser.
        val focalPx = p.rawWidth / 1.4
        val blurPx = if (p.front) 2.0 else 4.0
        val shakeLimit = when {
            p.still -> NIGHT_SHUTTER_NS.toDouble()
            !p.gyroTrusted || p.gyroRate <= 0.0 -> LOW_LIGHT_SHUTTER_NS.toDouble()
            else -> (blurPx / (p.gyroRate * focalPx) * 1e9).coerceIn(1e9 / 60, LOW_LIGHT_SHUTTER_NS.toDouble())
        }.toLong()

        var underUsed = 0f
        val (iso, exposure, gain) = if (p.manual != null) {
            Triple(p.manual.first, p.manual.second, 1f)
        } else {
            val product = p.autoIso.toDouble() * p.autoExposureNs * 2.0.pow(p.evCompensation)
            val underEv = when {
                p.clipped < 0f -> UNDER_EV
                p.clipped > 0.04f -> 1.6
                p.clipped > 0.01f -> 1.1
                p.clipped < 0.002f -> 0.4
                else -> UNDER_EV
            }
            underUsed = underEv.toFloat()
            val target = product * 2.0.pow(-underEv)
            // The gyroscope only shortens frames when there is light to spare: handheld, it never
            // pushes the digital gain past 2×.
            val base = min(SHUTTER_NS, shakeLimit)
            val atMaxIso = (target / (p.isoMax * if (p.still) 1.0 else DIGITAL_GAIN_FIRST)).toLong()
            val cap = atMaxIso.coerceIn(base, if (p.still) NIGHT_SHUTTER_NS else max(base, LOW_LIGHT_SHUTTER_NS))
            val exp = (target / p.isoMin).toLong().coerceIn(p.exposureMinNs, min(cap, p.exposureMaxNs))
            val i = (target / exp).roundToInt().coerceIn(p.isoMin, p.isoMax)
            Triple(i, exp, (product / (i.toDouble() * exp)).toFloat().coerceIn(1f, 64f))
        }

        val frames = if (p.hot) 4 else when {
            p.chosenFrames > 0 -> p.chosenFrames
            p.still && (gain >= 1.5f || exposure >= LOW_LIGHT_SHUTTER_NS) -> MAX_FRAMES
            gain >= 1.8f || iso >= p.isoMax * 0.9 -> 8
            else -> 6
        }.coerceIn(2, MAX_FRAMES)

        // Backlit scenes: two frames 2 EV shorter recover what is blown in the others.
        val shortExp = exposure / 4
        val bracket = p.manual == null && !p.still && p.clipped > 0.02f &&
            shortExp >= p.exposureMinNs && frames + 2 <= MAX_FRAMES
        return BurstPlan(
            iso, exposure, gain, frames, underUsed,
            shortFrames = if (bracket) 2 else 0,
            shortExposureNs = if (bracket) shortExp else 0,
        )
    }
}

/** EXIF helpers that need no Android classes. */
object ExifFormat {
    /** GPS coordinate: degrees, minutes and milliseconds of a second, as EXIF rationals. */
    fun dms(value: Double): String {
        val a = kotlin.math.abs(value)
        val d = a.toInt()
        val mFull = (a - d) * 60
        val m = mFull.toInt()
        val ms = ((mFull - m) * 60 * 1000).roundToInt()
        return "$d/1,$m/1,$ms/1000"
    }
}
