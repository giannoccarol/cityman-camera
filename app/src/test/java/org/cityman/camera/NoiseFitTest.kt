package org.cityman.camera

import org.cityman.camera.processing.MultiFrameProcessor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

class NoiseFitTest {
    /**
     * Builds the |difference| histograms the GPU would produce for a known noise model
     * (variance = a·x + b) and checks that the fit finds it again.
     */
    @Test
    fun recoversNoiseModel() {
        val a = 8e-4
        val b = 2e-5
        val counts = IntArray(16 * 64)
        val rnd = java.util.Random(1)
        for (bb in 0 until 16) {
            val x = ((bb + 0.5) / 16.0).let { it * it }
            val sigma = sqrt(a * x + b)
            repeat(20_000) {
                // difference of two noisy samples: sigma·√2
                val d = kotlin.math.abs(rnd.nextGaussian() * sigma * sqrt(2.0))
                val db = minOf((sqrt(d / 0.12) * 64).toInt(), 63)
                counts[bb * 64 + db]++
            }
        }
        val fit = MultiFrameProcessor.fitNoise(counts, 342)
        assertEquals(a, fit[0].toDouble(), a * 0.25)
        assertEquals(b, fit[1].toDouble(), b * 0.6)
    }

    @Test
    fun fallsBackWithoutData() {
        val fit = MultiFrameProcessor.fitNoise(IntArray(16 * 64), 100)
        assertEquals(1.2e-4, fit[0].toDouble(), 1e-6)
    }

    /** 10-bit sensor, black 64: one step in the units of the histogram. */
    private val quantum = 1.0 / (1023 - 64)

    /**
     * Two frames of the same scene as the sensor delivers them (whole steps), binned as the
     * NOISE shader does: brightness from the mean of the two, |difference| on a sqrt scale.
     * [maxSqrt]: the scene's brightness is spread evenly in sqrt up to this (0.35 = a dark scene).
     */
    private fun histogram(a: Double, b: Double, maxSqrt: Double, seed: Long, samples: Int = 600_000): IntArray {
        val rnd = java.util.Random(seed)
        val counts = IntArray(16 * 64)
        fun shot(s: Double): Double {
            val v = s + rnd.nextGaussian() * sqrt(a * s + b)
            val dn = (v / quantum + 64).roundToInt().coerceIn(0, 1023)
            return (dn - 64) * quantum
        }
        repeat(samples) {
            val s = (rnd.nextDouble() * maxSqrt).pow(2)
            val r = shot(s)
            val al = shot(s)
            val bb = (sqrt(max(0.5 * (r + al), 0.0)) * 16).toInt().coerceIn(0, 15)
            val db = minOf((sqrt(abs(al - r) / 0.12) * 64).toInt(), 63)
            counts[bb * 64 + db]++
        }
        return counts
    }

    private fun assertModel(a: Double, b: Double, fit: FloatArray, tolerance: Double) {
        for (x in doubleArrayOf(0.0, 0.002, 0.05, 0.3)) {
            val truth = a * x + b
            val got = fit[0] * x + fit[1]
            assertEquals("variance at x=$x (fit a=${fit[0]} b=${fit[1]})", truth, got, truth * tolerance)
        }
    }

    /**
     * Shadows with 0.6 to 2 steps of read noise, where the median of whole steps used to be off
     * by up to 2× either way and b ended at its lower limit.
     */
    @Test
    fun quantisedShadows() {
        val cases = listOf(
            Triple(100, 1.2e-4, 4e-7),      // read noise 0.6 steps
            Triple(200, 2.4e-4, 1.6e-6),    // 1.2 steps
            Triple(342, 4.1e-4, 1.5e-6),    // 1.2 steps: b used to come out as 1e-7
            Triple(342, 4.1e-4, 4.7e-6),    // 2.1 steps
        )
        for ((iso, a, b) in cases) {
            for (maxSqrt in doubleArrayOf(0.85, 0.35)) {
                val fit = MultiFrameProcessor.fitNoise(histogram(a, b, maxSqrt, seed = 7), iso, quantum)
                assertModel(a, b, fit, 0.25)
            }
        }
    }

    /** Noise far below one step: every difference is zero, and b is the quantisation noise. */
    @Test
    fun belowOneStep() {
        val fit = MultiFrameProcessor.fitNoise(histogram(1e-6, 1e-9, 0.85, seed = 3), 100, quantum)
        assertEquals(quantum * quantum / 12, fit[1].toDouble(), 1e-9)
    }

    /** The flat-area histogram is used when it kept enough samples, the full one otherwise. */
    @Test
    fun flatAreasWhenEnough() {
        val a = 4.1e-4
        val b = 1.5e-6
        val flat = histogram(a, b, 0.85, seed = 11)
        // The full histogram also holds texture residue: three times the noise.
        val all = histogram(9 * a, 9 * b, 0.85, seed = 12)
        for (i in all.indices) all[i] += flat[i]
        assertModel(a, b, MultiFrameProcessor.fitNoise(all + flat, 342, quantum), 0.25)
        // Hardly any flat sample: the fit must not rest on them.
        val few = IntArray(flat.size) { flat[it] / 40 }
        val fit = MultiFrameProcessor.fitNoise(all + few, 342, quantum)
        assertTrue("fit a=${fit[0]} should see the texture too", fit[0] > 2 * a)
    }

    @Test
    fun inverseErrorFunction() {
        assertEquals(0.4769362762, MultiFrameProcessor.erfInv(0.5), 1e-6)
        assertEquals(0.1791434546, MultiFrameProcessor.erfInv(0.2), 1e-6)
        assertEquals(1.1630871537, MultiFrameProcessor.erfInv(0.9), 1e-6)
        assertEquals(1.6449763571, MultiFrameProcessor.erfInv(0.98), 1e-5)
    }
}
