package org.cityman.camera

import org.cityman.camera.camera.BurstPlanner
import org.cityman.camera.camera.ExifFormat
import org.cityman.camera.camera.PlanInput
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.pow

/** Burst plans for the rear IMX230 (manual ISO 42–342, 12.6 µs – 0.83 s). */
class BurstPlannerTest {
    private val rear = PlanInput(
        isoMin = 42, isoMax = 342, exposureMinNs = 12_651L, exposureMaxNs = 828_982_077L,
        autoIso = 100, autoExposureNs = 1_000_000L,
    )

    private fun digitalGain(gain: Float, underEv: Float) = gain / 2.0.pow(underEv.toDouble())

    @Test
    fun daylightUsesLowIsoShortFrames() {
        val p = BurstPlanner.plan(rear.copy(clipped = 0.001f))
        assertEquals(42, p.iso)
        assertTrue(p.exposureNs <= BurstPlanner.SHUTTER_NS)
        assertEquals(6, p.frames)
        assertEquals(0, p.shortFrames)
        assertEquals(0.4f, p.underEv, 1e-6f)
        assertEquals(1.0, digitalGain(p.gain, p.underEv), 0.05)
    }

    @Test
    fun dimRoomHandheldCapsShutterAndUsesDigitalGain() {
        val p = BurstPlanner.plan(rear.copy(autoIso = 1600, autoExposureNs = 66_000_000L, clipped = 0.003f))
        assertEquals(342, p.iso)
        assertTrue(p.exposureNs <= BurstPlanner.LOW_LIGHT_SHUTTER_NS)
        assertEquals(8, p.frames)
        assertTrue(digitalGain(p.gain, p.underEv) <= 2.05)
    }

    @Test
    fun tripodGetsLongFramesAndTwelveOfThem() {
        val p = BurstPlanner.plan(rear.copy(autoIso = 1600, autoExposureNs = 66_000_000L, clipped = 0.003f, still = true))
        assertEquals(12, p.frames)
        assertTrue(p.exposureNs > BurstPlanner.LOW_LIGHT_SHUTTER_NS)
        assertTrue(p.exposureNs <= BurstPlanner.NIGHT_SHUTTER_NS)
        assertTrue(digitalGain(p.gain, p.underEv) < 1.05)
    }

    @Test
    fun shakyHandShortensFramesButKeepsDigitalGainBounded() {
        val p = BurstPlanner.plan(
            rear.copy(autoIso = 400, autoExposureNs = 33_000_000L, gyroTrusted = true, gyroRate = 0.2),
        )
        assertTrue("exposure ${p.exposureNs}", p.exposureNs <= 16_700_000L)
        assertTrue(digitalGain(p.gain, p.underEv) <= 2.0)
    }

    @Test
    fun backlitSceneAddsHighlightFrames() {
        val p = BurstPlanner.plan(rear.copy(autoIso = 100, autoExposureNs = 10_000_000L, clipped = 0.05f))
        assertEquals(1.6f, p.underEv, 1e-6f)
        assertEquals(2, p.shortFrames)
        assertEquals(p.exposureNs / 4, p.shortExposureNs)
    }

    @Test
    fun manualSettingsAreKept() {
        val p = BurstPlanner.plan(rear.copy(manual = 200 to 10_000_000L, clipped = 0.05f))
        assertEquals(200, p.iso)
        assertEquals(10_000_000L, p.exposureNs)
        assertEquals(1f, p.gain, 0f)
        assertEquals(0, p.shortFrames)
    }

    @Test
    fun hotPhoneTakesFourFrames() {
        assertEquals(4, BurstPlanner.plan(rear.copy(hot = true, chosenFrames = 8)).frames)
    }

    @Test
    fun chosenFramesWin() {
        assertEquals(4, BurstPlanner.plan(rear.copy(chosenFrames = 4)).frames)
    }

    @Test
    fun gpsRationals() {
        assertEquals("45/1,30/1,0/1000", ExifFormat.dms(45.5))
        assertEquals("12/1,20/1,44160/1000", ExifFormat.dms(-12.3456))
    }
}
