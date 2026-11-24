package org.cityman.camera

import org.cityman.camera.camera.VideoQuality
import org.cityman.camera.camera.VideoQuality.FHD30
import org.cityman.camera.camera.VideoQuality.FHD60
import org.cityman.camera.camera.VideoQuality.HD30
import org.cityman.camera.camera.VideoQuality.HD60
import org.cityman.camera.camera.VideoQuality.UHD30
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class VideoQualityTest {
    @Test
    fun wantedWhenOffered() = assertEquals(FHD60, VideoQuality.pick(listOf(HD30, FHD30, FHD60), FHD60))

    @Test
    fun sameFrameRateSmallerSizeFirst() = assertEquals(HD60, VideoQuality.pick(listOf(HD30, HD60, FHD30), FHD60))

    @Test
    fun thenThirtyAtTheSameSize() = assertEquals(FHD30, VideoQuality.pick(listOf(HD30, FHD30), FHD60))

    @Test
    fun fourKFallsBackToTheLargestThirty() = assertEquals(FHD30, VideoQuality.pick(listOf(HD30, HD60, FHD30), UHD30))

    @Test
    fun smallWantedTakesTheSmallestOffered() = assertEquals(FHD30, VideoQuality.pick(listOf(FHD30, UHD30), HD30))

    @Test
    fun nothingOffered() = assertNull(VideoQuality.pick(emptyList(), FHD30))

    @Test
    fun unknownPreferenceIsTheDefault() = assertEquals(FHD30, VideoQuality.fromName("bogus"))
}
