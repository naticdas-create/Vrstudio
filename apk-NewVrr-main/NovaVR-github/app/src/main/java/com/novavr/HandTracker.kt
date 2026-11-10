package com.novavr

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.hypot

/**
 * pts = 21 pontos (x,y normalizados) da mao; z = profundidade relativa de cada ponto (menor = mais perto da camera).
 * left = indice do "slot" da mao (true = slot 0, false = slot 1): fixo por mao, nao depende do rotulo Esquerda/Direita.
 * pinch = pinca do indicador (estado confirmado); strength = progresso da pinca de 0 a 1;
 * mid/ring/pinky = pinca do dedo medio/anelar/minimo; conf = confianca do rastreio (0 a 1).
 */
class Hand(
    val left: Boolean,
    val pts: FloatArray,
    val pinch: Boolean,
    val strength: Float = 0f,
    val mid: Boolean = false,
    val ring: Boolean = false,
    val pinky: Boolean = false,
    val conf: Float = 1f,
    val z: FloatArray = FloatArray(21)
)

/**
 * Filtro One Euro: parado = bem firme (sem tremor), em movimento = acompanha rapido (sem atraso).
 * minCut baixo = mais firme; beta alto = mais rapido ao mover.
 */
private class OneEuro(private val minCut: Float, private val beta: Float) {
    private var xp = 0f
    private var dxp = 0f
    private var on = false
    private fun alpha(cut: Float, dt: Float): Float {
        val tau = 1f / (2f * PI.toFloat() * cut)
        return 1f / (1f + tau / dt)
    }
    fun reset() { on = false }
    fun filter(x: Float, dt: Float): Float {
        if (!on) { on = true; xp = x; dxp = 0f; return x }
        val dx = (x - xp) / dt
        dxp += alpha(1f, dt) * (dx - dxp)
        xp += alpha(minCut + beta * abs(dxp), dt) * (x - xp)
        return xp
    }
}

class HandTracker(ctx: Context) {
    private val lm: HandLandmarker = HandLandmarker.createFromOptions(
        ctx,
        HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build())
            .setNumHands(2)
            .setMinHandDetectionConfidence(.4f)
            .setMinHandPresenceConfidence(.4f)
            .setMinTrackingConfidence(.4f)
            .setRunningMode(RunningMode.VIDEO)
            .build()
    )
    private val pin = BooleanArray(2)
    private val pinM = BooleanArray(2)
    private val pinR = BooleanArray(2)
    private val pinP = BooleanArray(2)
    private val sm = Array(2) { FloatArray(42) }              // pontos filtrados (saida)
    private val smz = Array(2) { FloatArray(21) }             // profundidades filtradas
    private val flt = Array(2) { Array(42) { OneEuro(1.4f, 30f) } }
    private val lastH = arrayOfNulls<Hand>(2)                 // ultima mao vista em cada slot
    private val lastT = LongArray(2)
    private val slotX = FloatArray(2); private val slotY = FloatArray(2)   // punho de cada slot (para nao trocar de mao)
    private var lastDet = 0L

    fun detect(b: Bitmap): List<Hand> {
        val now = SystemClock.uptimeMillis()
        val dt = if (lastDet == 0L) .033f else ((now - lastDet) / 1000f).coerceIn(.008f, .1f)
        lastDet = now
        val r = lm.detectForVideo(BitmapImageBuilder(b).build(), now)
        val lms = r.landmarks()
        val hds = r.handedness()
        val n = minOf(lms.size, 2)

        // cada mao fica sempre no mesmo slot, pela posicao do punho (o rotulo Esquerda/Direita do modelo oscila)
        val wx = FloatArray(n) { lms[it][0].x() }
        val wy = FloatArray(n) { lms[it][0].y() }
        fun live(k: Int) = lastT[k] != 0L && now - lastT[k] < 700L
        fun dist(i: Int, k: Int) = hypot(wx[i] - slotX[k], wy[i] - slotY[k])
        val slot = IntArray(n)
        if (n == 1) {
            val c0 = if (live(0)) dist(0, 0) else 9f
            val c1 = if (live(1)) dist(0, 1) else 9f
            slot[0] = if (c0 < 9f || c1 < 9f) (if (c0 <= c1) 0 else 1)
            else (if (hds.getOrNull(0)?.firstOrNull()?.categoryName() == "Left") 0 else 1)
        } else if (n == 2) {
            if (live(0) && live(1)) {
                val straight = dist(0, 0) + dist(1, 1); val swap = dist(0, 1) + dist(1, 0)
                if (straight <= swap) { slot[0] = 0; slot[1] = 1 } else { slot[0] = 1; slot[1] = 0 }
            } else if (wx[0] <= wx[1]) { slot[0] = 0; slot[1] = 1 } else { slot[0] = 1; slot[1] = 0 }
        }

        val out = ArrayList<Hand>()
        val seen = BooleanArray(2)
        for (i in 0 until n) {
            val h = lms[i]
            val k = slot[i]
            val conf = hds.getOrNull(i)?.firstOrNull()?.score() ?: 1f
            seen[k] = true
            slotX[k] = wx[i]; slotY[k] = wy[i]
            fun d(a: Int, c: Int) = hypot(h[a].x() - h[c].x(), h[a].y() - h[c].y())
            val palm = d(0, 9).coerceAtLeast(1e-3f)
            val rI = d(4, 8) / palm
            val rM = d(4, 12) / palm
            val rR = d(4, 16) / palm
            val rP = d(4, 20) / palm
            pin[k] = if (pin[k]) rI < 0.65f else rI < 0.4f
            // os outros dedos so contam se forem o mais proximo do polegar (evita confundir com a pinca do indicador)
            pinM[k] = if (pinM[k]) rM < 0.65f else (rM < 0.4f && rM < rI - 0.05f)
            pinR[k] = if (pinR[k]) rR < 0.65f else (rR < 0.4f && rR < rI - 0.05f && rR < rM - 0.05f)
            pinP[k] = if (pinP[k]) rP < 0.65f else (rP < 0.4f && rP < rI - 0.05f && rP < rM - 0.05f && rP < rR - 0.05f)
            val strength = ((0.7f - rI) / 0.5f).coerceIn(0f, 1f)
            val sp = sm[k]
            val sz = smz[k]
            for (j in 0 until 21) {
                sp[2 * j] = flt[k][2 * j].filter(h[j].x(), dt)
                sp[2 * j + 1] = flt[k][2 * j + 1].filter(h[j].y(), dt)
                sz[j] += (h[j].z() - sz[j]) * .5f
            }
            val hand = Hand(k == 0, sp.copyOf(), pin[k], strength, pinM[k] && !pin[k], pinR[k] && !pin[k], pinP[k] && !pin[k], conf, sz.copyOf())
            out.add(hand)
            lastH[k] = hand; lastT[k] = now
        }
        for (k in 0..1) {
            if (seen[k]) continue
            for (f in flt[k]) f.reset()
            val h = lastH[k]
            val age = now - lastT[k]
            val grace = if (h != null && h.pinch) 350L else 200L
            if (h == null || age >= grace) {            // passou o tempo: solta tudo
                pin[k] = false; pinM[k] = false; pinR[k] = false; pinP[k] = false
                continue
            }
            // perdeu o rastreio por um instante: a mao some aos poucos (como no Quest); a pinca fica presa um pouco mais
            // para o arraste nao soltar no meio quando o modelo falha 1 ou 2 quadros
            val c = if (h.pinch) maxOf(h.conf, .6f) else h.conf * (1f - age / 200f)
            out.add(Hand(h.left, h.pts, h.pinch, h.strength, false, false, false, c, h.z))
        }
        return out
    }

    fun close() = lm.close()
}
