package org.cityman.camera.camera

import android.hardware.camera2.params.ColorSpaceTransform
import android.hardware.camera2.params.LensShadingMap
import android.hardware.camera2.params.RggbChannelVector
import android.media.Image
import org.cityman.camera.processing.BurstSetup
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream

/**
 * Developer option: writes a burst's RAW frames and the metadata the merge used, so the pipeline
 * can be reproduced and tuned on a computer (`tools/burst-lab`). Frames are RAW16 little endian,
 * tightly packed (width × 2 bytes per row).
 */
internal object BurstDump {
    fun writeFrame(dir: File, index: Int, image: Image) {
        val plane = image.planes[0]
        val buf = plane.buffer.duplicate()
        val rowBytes = image.width * 2
        val row = ByteArray(rowBytes)
        FileOutputStream(File(dir, String.format("frame_%02d.raw16", index))).use { out ->
            for (y in 0 until image.height) {
                buf.position(y * plane.rowStride)
                buf.get(row, 0, rowBytes)
                out.write(row)
            }
        }
    }

    fun writeMeta(
        dir: File,
        setup: BurstSetup,
        plan: CameraController.ShotInfo,
        gains: RggbChannelVector,
        ccm: ColorSpaceTransform,
        shading: LensShadingMap?,
        received: Int,
    ) {
        val json = JSONObject()
        json.put("width", setup.width)
        json.put("height", setup.height)
        json.put("chanAt", JSONArray(setup.chanAt.toList()))
        json.put("black", JSONArray(setup.black.map { it.toDouble() }))
        json.put("white", setup.white.toDouble())
        json.put("rotation", setup.rotation)
        json.put("crop", setup.crop)
        json.put("frames", setup.frames)
        json.put("received", received)
        json.put("shortFrames", setup.shortFrames)
        json.put("shortScale", setup.shortScale.toDouble())
        json.put("gain", setup.gain.toDouble())
        json.put("underEv", setup.underEv.toDouble())
        json.put("zoom", setup.zoom.toDouble())
        json.put("iso", plan.iso)
        json.put("exposureNs", plan.exposureNs)
        json.put("options", JSONObject().apply {
            put("denoise", setup.options.denoise)
            put("vivid", setup.options.vivid)
            put("localTone", setup.options.localTone)
            put("chromaDenoise", setup.options.chromaDenoise)
            put("sharpen", setup.options.sharpen)
        })
        json.put("gains", JSONArray(listOf(gains.red, gains.greenEven, gains.greenOdd, gains.blue).map { it.toDouble() }))
        val m = JSONArray()
        for (r in 0 until 3) for (c in 0 until 3) m.put(ccm.getElement(c, r).toDouble())
        json.put("ccm", m)
        if (shading != null) {
            val data = JSONArray()
            for (r in 0 until shading.rowCount) for (c in 0 until shading.columnCount) for (ch in 0 until 4) {
                data.put(shading.getGainFactor(ch, c, r).toDouble())
            }
            json.put("lsc", JSONObject().apply {
                put("cols", shading.columnCount)
                put("rows", shading.rowCount)
                put("rggb", data)
            })
        }
        File(dir, "meta.json").writeText(json.toString(1))
    }
}
