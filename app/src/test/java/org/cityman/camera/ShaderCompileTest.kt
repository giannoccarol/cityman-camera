package org.cityman.camera

import org.cityman.camera.processing.Shaders
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * Compiles every compute shader of the multi-frame pipeline with glslangValidator (the Khronos
 * reference compiler), when it is installed: a typo in a shader otherwise only shows on a phone,
 * as a multi-frame photo that fails. Skipped where the tool is missing.
 */
class ShaderCompileTest {
    private fun validator(): String? = listOf("glslangValidator", "glslang").firstOrNull { name ->
        runCatching { ProcessBuilder(name, "--version").redirectErrorStream(true).start().apply { inputStream.readBytes() }.waitFor() == 0 }.getOrDefault(false)
    }

    @Test
    fun computeShadersCompile() {
        val tool = validator()
        assumeTrue("glslangValidator not installed", tool != null)
        val shaders = Shaders::class.java.declaredMethods
            .filter { it.name.startsWith("get") && it.returnType == String::class.java && it.parameterCount == 0 }
            .associate { it.name.removePrefix("get") to it.invoke(Shaders) as String }
            .filterValues { "local_size_x" in it }
        assertTrue("no shaders found", shaders.size >= 15)
        val dir = java.nio.file.Files.createTempDirectory("arco-shaders").toFile()
        try {
            for ((name, source) in shaders) {
                val file = File(dir, "$name.comp").apply { writeText(source) }
                val p = ProcessBuilder(tool, "-S", "comp", file.path).redirectErrorStream(true).start()
                val out = p.inputStream.bufferedReader().readText()
                assertEquals("shader $name:\n$out", 0, p.waitFor())
            }
        } finally {
            dir.deleteRecursively()
        }
    }
}
