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
import kotlin.math.sqrt

/**
 * pts = 21 pontos (x,y normalizados) da mao.
 * slot = 0 ou 1: identidade ESTAVEL da mao (casada por posicao, nao pelo rotulo esquerda/direita, que o MediaPipe troca toda hora).
 * pinch = pinca do indicador (estado confirmado); strength = progresso da pinca de 0 a 1;
 * mid/ring/pinky = pinca dos outros dedos; conf = 1 com mao detectada, cai a 0 durante uma perda curta;
 * held = true quando o rastreio falhou e a mao esta sendo mantida (pinca continua valendo, o arrasto nao solta).
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
    val held: Boolean = false,
    val slot: Int = 0
)

/** Filtro One Euro: parado = firme (sem tremor), em movimento = acompanha rapido. */
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
            .setMinHandDetectionConfidence(0.35f)      // padrao 0.5: perdia a mao com facilidade
            .setMinHandPresenceConfidence(0.35f)
            .setMinTrackingConfidence(0.35f)
            .setRunningMode(RunningMode.VIDEO)
            .build()
    )

    private val N = 2
    private val HOLD_MS = 320L                                   // perdeu o rastreio: segura a mao (e a pinca) por este tempo
    private val pin = BooleanArray(N)
    private val pinM = BooleanArray(N)
    private val pinR = BooleanArray(N)
    private val pinP = BooleanArray(N)
    private val rIs = FloatArray(N)                              // razao da pinca suavizada
    private val sm = Array(N) { FloatArray(42) }
    private val flt = Array(N) { Array(42) { OneEuro(1.4f, 30f) } }
    private val active = BooleanArray(N)
    private val lastHand = arrayOfNulls<Hand>(N)
    private val lastSeen = LongArray(N)
    private val wrist = Array(N) { FloatArray(2) }
    private var lastDet = 0L

    fun detect(b: Bitmap): List<Hand> {
        val now = SystemClock.uptimeMillis()
        val dt = if (lastDet == 0L) .033f else ((now - lastDet) / 1000f).coerceIn(.008f, .1f)
        lastDet = now
        val r = lm.detectForVideo(BitmapImageBuilder(b).build(), now)
        val lms = r.landmarks()
        val wls = r.worldLandmarks()
        val hds = r.handedness()
        val bw = b.width.toFloat(); val bh = b.height.toFloat()

        // descarta deteccao duplicada (duas maos "fantasma" no mesmo lugar)
        val keep = ArrayList<Int>()
        for (i in 0 until lms.size) {
            var dup = false
            for (j in keep) if (hypot(lms[i][0].x() - lms[j][0].x(), lms[i][0].y() - lms[j][0].y()) < .05f) dup = true
            if (!dup && keep.size < N) keep.add(i)
        }
        val n = keep.size

        // casa cada deteccao com o "slot" da mao mais proxima no quadro anterior
        val assign = IntArray(n) { -1 }
        val taken = BooleanArray(N)
        val cand = ArrayList<Triple<Float, Int, Int>>()
        for (a in 0 until n) for (s in 0 until N) {
            if (!active[s]) continue
            val h = lms[keep[a]]
            cand.add(Triple(hypot(h[0].x() - wrist[s][0], h[0].y() - wrist[s][1]), a, s))
        }
        cand.sortBy { it.first }
        for ((dist, a, s) in cand) {
            if (dist < .4f && assign[a] < 0 && !taken[s]) { assign[a] = s; taken[s] = true }
        }
        val fresh = BooleanArray(N)
        for (a in 0 until n) if (assign[a] < 0) {
            val s = (0 until N).firstOrNull { !taken[it] && !active[it] }
                ?: (0 until N).filter { !taken[it] }.minByOrNull { lastSeen[it] }
                ?: continue
            assign[a] = s; taken[s] = true; fresh[s] = true
        }

        val out = ArrayList<Hand>()
        val seen = BooleanArray(N)
        for (a in 0 until n) {
            val i = keep[a]
            val s = assign[a]
            if (s < 0) continue
            val h = lms[i]
            val w3 = wls.getOrNull(i)
            val cat = hds.getOrNull(i)?.firstOrNull()
            // o MediaPipe assume imagem espelhada (selfie); a camera traseira nao e: inverte o rotulo
            val left = cat?.categoryName() == "Right"
            if (fresh[s]) {
                for (f in flt[s]) f.reset()
                pin[s] = false; pinM[s] = false; pinR[s] = false; pinP[s] = false; rIs[s] = 1f
            }
            // distancia: 3D em metros (world landmarks) quando existe, senao 2D corrigida pela proporcao da imagem
            fun dist(p: Int, q: Int): Float {
                if (w3 != null) {
                    val dx = w3[p].x() - w3[q].x(); val dy = w3[p].y() - w3[q].y(); val dz = w3[p].z() - w3[q].z()
                    return sqrt(dx * dx + dy * dy + dz * dz)
                }
                return hypot((h[p].x() - h[q].x()) * bw, (h[p].y() - h[q].y()) * bh)
            }
            val palm = dist(0, 9).coerceAtLeast(1e-3f)
            val rI = dist(4, 8) / palm
            val rM = dist(4, 12) / palm
            val rR = dist(4, 16) / palm
            val rP = dist(4, 20) / palm
            rIs[s] = if (fresh[s]) rI else rIs[s] + (rI - rIs[s]) * .65f
            val ri = rIs[s]
            pin[s] = if (pin[s]) ri < .60f else ri < .38f
            // os outros dedos so contam com o indicador aberto e se forem os mais proximos do polegar
            pinM[s] = if (pinM[s]) rM < .60f else (rM < .36f && rM < rI - .08f && rI > .6f)
            pinR[s] = if (pinR[s]) rR < .60f else (rR < .36f && rR < rI - .08f && rR < rM - .05f && rI > .6f)
            pinP[s] = if (pinP[s]) rP < .60f else (rP < .36f && rP < rI - .08f && rP < rM - .05f && rP < rR - .05f && rI > .6f)
            val strength = ((.78f - ri) / .42f).coerceIn(0f, 1f)
            val sp = sm[s]
            for (j in 0 until 21) {
                sp[2 * j] = flt[s][2 * j].filter(h[j].x(), dt)
                sp[2 * j + 1] = flt[s][2 * j + 1].filter(h[j].y(), dt)
            }
            val hand = Hand(left, sp.copyOf(), pin[s], strength, pinM[s] && !pin[s], pinR[s] && !pin[s], pinP[s] && !pin[s], 1f, false, s)
            out.add(hand)
            seen[s] = true; active[s] = true
            lastHand[s] = hand; lastSeen[s] = now
            wrist[s][0] = h[0].x(); wrist[s][1] = h[0].y()
        }
        // perdeu o rastreio de uma mao: segura a ultima pose (e a pinca) por um instante, sumindo aos poucos
        for (s in 0 until N) {
            if (seen[s]) continue
            val lh = lastHand[s] ?: continue
            val age = now - lastSeen[s]
            if (age <= HOLD_MS) {
                val f = (1f - age / HOLD_MS.toFloat()).coerceIn(0f, 1f)
                out.add(Hand(lh.left, lh.pts, lh.pinch, lh.strength, false, false, false, f, true, s))
            } else if (age > 900L) {
                active[s] = false; lastHand[s] = null
            }
        }
        return out
    }

    fun close() = lm.close()
}
