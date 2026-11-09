package org.cityman.camera.camera

import android.app.ActivityManager
import android.content.Context
import android.os.Build

/**
 * What the phone itself (not one camera) offers, read once at start. Arco runs from Android 11
 * on, on GPUs from OpenGL ES 3.0: features that need more turn themselves off.
 */
object DeviceSupport {
    /** OpenGL ES version the system reports, as 0xMMMMmmmm (0x00030001 = 3.1). */
    @Volatile
    var glesVersion: Int = 0
        private set

    /** Set when the GPU failed to run the multi-frame shaders although it claims ES 3.1. */
    @Volatile
    var computeFailed: Boolean = false

    /**
     * Multi-frame merging runs on compute shaders, which need OpenGL ES 3.1. An unknown version
     * (no `ro.opengles.version`) is tried, and [computeFailed] turns it off if the GPU refuses.
     */
    val computeShaders: Boolean get() = (glesVersion == 0 || glesVersion >= GLES_3_1) && !computeFailed

    /** Wallpaper (Material You) colours exist from Android 12. */
    val dynamicColor: Boolean get() = Build.VERSION.SDK_INT >= Build.VERSION_CODES.S

    val glesLabel: String get() = if (glesVersion == 0) "?" else "${glesVersion ushr 16}.${glesVersion and 0xffff}"

    fun init(context: Context) {
        if (glesVersion != 0) return
        glesVersion = runCatching {
            context.getSystemService(ActivityManager::class.java).deviceConfigurationInfo.reqGlEsVersion
        }.getOrDefault(0)
    }

    private const val GLES_3_1 = 0x30001
}
