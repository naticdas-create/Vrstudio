package com.novavr

import android.app.Activity
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.RadialGradient
import android.graphics.RectF
import android.graphics.Shader
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.net.Uri
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.GLUtils
import android.opengl.Matrix as GLM
import android.view.MotionEvent
import android.view.SoundEffectConstants
import android.view.Surface
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.Locale
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.*

private fun dot(a: FloatArray, b: FloatArray) = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
private fun cross(a: FloatArray, b: FloatArray) =
    floatArrayOf(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
private fun norm(a: FloatArray): FloatArray {
    val l = sqrt(dot(a, a)).coerceAtLeast(1e-6f)
    return floatArrayOf(a[0] / l, a[1] / l, a[2] / l)
}

class Btn(val id: String, val r: RectF)

/** Janela 3D: posicao e orientacao fixas no mundo. Conteudo + barra de titulo. */
class Win(val id: String, val w: Float, val h: Float, val bw: Int, val bh: Int, val hasTitle: Boolean = true) {
    var title = ""
    val pos = floatArrayOf(0f, 0f, -2f)
    var r = floatArrayOf(1f, 0f, 0f)
    var u = floatArrayOf(0f, 1f, 0f)
    var n = floatArrayOf(0f, 0f, 1f)
    var scale = 1f
    var visible = true
    var bmp: Bitmap? = null
    var tex = 0
    var dirty = true
    var uScale = 1f
    val btns = ArrayList<Btn>()
    val tbw = (w * 600).toInt()
    val tbh = 60
    var tBmp: Bitmap? = null
    var tTex = 0
    var tDirty = true
    val tbBtns = ArrayList<Btn>()
    var sd = 2f      // distancia ao abrir
    var sy = 0f      // altura ao abrir
    var sa = 0f      // angulo ao abrir (graus)
    var anim = 0f                // 0..1: animacao de abrir/fechar
    var glow = 0f                // 0..1: brilho da borda quando apontada
    var rad = .06f               // raio dos cantos (metros)
    val thM get() = 0.1f * scale
}

private const val CURVE_R = 3.5f   // raio da curvatura das janelas (metros), como no Quest

private const val VS_W = """
attribute vec2 aPos; uniform mat4 uMVP; uniform vec4 uUV; uniform float uBend; varying vec2 vUV; varying vec2 vLocal; varying float vDepth;
void main(){ vLocal = aPos;
  vec4 pos = uMVP * vec4(aPos.x, aPos.y, aPos.x * aPos.x * uBend, 1.0);
  gl_Position = pos; vDepth = pos.w;
  vUV = uUV.xy + vec2(aPos.x + 0.5, 0.5 - aPos.y) * uUV.zw; }"""
private const val FS_W = """
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform sampler2D uTex; uniform float uA; uniform vec2 uQuad; uniform vec2 uBox; uniform float uRad; uniform float uGlow; uniform float uShadow;
uniform sampler2D uDepth; uniform vec4 uVp; uniform vec4 uCrop; uniform float uOcc;
varying vec2 vUV; varying vec2 vLocal; varying float vDepth;
float sdBox(vec2 p, vec2 b, float r){ vec2 q = abs(p) - b + vec2(r); return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r; }
void main(){
  vec2 p = vLocal * uQuad;
  float d = sdBox(p, uBox * 0.5, uRad);
  float occ = 0.0;
  if (uOcc > 0.5) {
    vec2 sp = (gl_FragCoord.xy - uVp.xy) / uVp.zw;
    vec2 cuv = vec2(uCrop.x + sp.x * uCrop.z, uCrop.y + (1.0 - sp.y) * uCrop.w);
    float od = texture2D(uDepth, cuv).r * 8.0;
    occ = smoothstep(0.25, 0.65, vDepth - od);
  }
  if (uShadow > 0.5) {
    float a = (1.0 - smoothstep(-0.01, 0.075, d)) * 0.5 * uA * (1.0 - occ);
    gl_FragColor = vec4(0.0, 0.0, 0.0, a);
    return;
  }
  float aa = 1.0 - smoothstep(-0.003, 0.003, d);
  vec4 c = texture2D(uTex, vUV);
  float rim = smoothstep(-0.014, -0.002, d);
  float k = rim * (0.16 + 0.55 * uGlow);
  vec3 rc = vec3(0.90, 0.92, 1.0);
  c.rgb = c.rgb * (1.0 - k) + rc * k * max(c.a, 0.5);
  c.a = max(c.a, k * 0.5);
  gl_FragColor = c * aa * uA * (1.0 - occ);
}"""
private const val VS_Q = """
attribute vec2 aPos; uniform mat4 uMVP; uniform vec4 uUV; varying vec2 vUV;
void main(){ gl_Position = uMVP * vec4(aPos, 0.0, 1.0);
  vUV = uUV.xy + vec2(aPos.x + 0.5, 0.5 - aPos.y) * uUV.zw; }"""
private const val FS_Q = """
precision mediump float; uniform sampler2D uTex; uniform float uA; varying vec2 vUV;
void main(){ gl_FragColor = texture2D(uTex, vUV) * uA; }"""
private const val VS_D = """
attribute vec2 aPos; varying vec2 vUV;
void main(){ gl_Position = vec4(aPos * 2.0, 0.0, 1.0); vUV = aPos + 0.5; }"""
private const val FS_D = """
precision highp float; uniform sampler2D uTex; uniform float uK1; uniform float uK2; uniform float uZ; varying vec2 vUV;
void main(){
  float eye = step(0.5, vUV.x);
  vec2 l = vec2((vUV.x - eye * 0.5) * 2.0, vUV.y);
  vec2 c = (l - 0.5) * 2.0;
  float r2 = dot(c, c) * 0.5;
  vec2 s = c * (1.0 + uK1 * r2 + uK2 * r2 * r2) * uZ;
  if (abs(s.x) > 1.0 || abs(s.y) > 1.0) { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }
  vec2 uv = vec2((s.x * 0.5 + 0.5) * 0.5 + eye * 0.5, s.y * 0.5 + 0.5);
  gl_FragColor = texture2D(uTex, uv);
}"""

/**
 * LUNARO VR - motor 3D (OpenGL ES 2): estereo com parallax real, correcao de lente,
 * janelas world-locked, camera de fundo, maos com laser e pinca.
 */
class VrGlView(private val act: Activity, private val browser: BrowserSurface, private val onExit: () -> Unit) :
    GLSurfaceView(act), SensorEventListener, GLSurfaceView.Renderer {

    private class Frame(val bmp: Bitmap, val hands: List<Hand>)
    @Volatile private var frame: Frame? = null
    /** Camera + maos do MESMO quadro (a mao desenhada fica alinhada com a imagem). */
    fun pushFrame(b: Bitmap, hs: List<Hand>) { frame = Frame(b, hs) }
    private var curFr: Frame? = null                           // quadro fixado no inicio de cada desenho
    private val cam: Bitmap? get() = curFr?.bmp
    private val hands: List<Hand> get() = curFr?.hands ?: emptyList()

    // ---------- janelas ----------
    private val browserWin = Win("web", 2.2f, 2.2f * browser.h / browser.w, browser.w, browser.h).also { it.title = "Navegador"; it.sd = 2.2f; it.sy = .2f; it.rad = .07f }
    private val dock = Win("dock", 1.6f, .16f, 1280, 128, false).also { it.sd = 1.8f; it.sy = -1.0f; it.rad = .08f }
    private val addr = Win("addr", 2.2f, .14f, 1760, 112, false).also { it.sd = 2.2f; it.sy = .93f; it.rad = .07f }
    private val apps = Win("apps", 1.3f, .9f, 1040, 720).also { it.title = "Biblioteca"; it.sd = 2.0f; it.sy = .1f; it.sa = -38f; it.rad = .07f }
    private val kb = Win("kb", 1.6f, .78f, 1280, 624).also { it.title = "Teclado"; it.sd = 1.6f; it.sy = -.95f; it.rad = .06f }
    private val cfg = Win("cfg", 1f, .8f, 1000, 800).also { it.title = "Configuracoes"; it.sd = 2.0f; it.sy = .1f; it.sa = 38f; it.rad = .06f }
    private val wins = listOf(browserWin, addr, dock, apps, kb, cfg)

    private class App(val id: String, val name: String, val url: String, val color: Int, val glyph: String)
    private val appList = listOf(
        App("yt", "YouTube", "https://www.youtube.com/", 0xFFE5173F.toInt(), "YT"),
        App("nf", "Netflix", "https://www.netflix.com/", 0xFFB20710.toInt(), "N"),
        App("sp", "Spotify", "https://open.spotify.com/", 0xFF1DB954.toInt(), "SP"),
        App("g", "Google", "https://www.google.com/", 0xFF2F7DE1.toInt(), "G"),
        App("dp", "Disney+", "https://www.disneyplus.com/", 0xFF113CCF.toInt(), "D+"),
        App("tw", "Twitch", "https://www.twitch.tv/", 0xFF7C5CFF.toInt(), "TW"),
        App("rb", "Roblox", "https://www.roblox.com/", 0xFFD62B2B.toInt(), "RBX"),
        App("pv", "Prime Video", "https://www.primevideo.com/", 0xFF1A98FF.toInt(), "PV"),
        App("gp", "Globoplay", "https://globoplay.globo.com/", 0xFFEF4444.toInt(), "GP"))
    private val iconBmps = HashMap<String, Bitmap>()
    private val ip = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
    private fun loadIcons() {
        for ((id, b64) in IconData.DATA) {                       // icones embutidos no codigo
            try {
                val bytes = android.util.Base64.decode(b64, android.util.Base64.DEFAULT)
                BitmapFactory.decodeByteArray(bytes, 0, bytes.size)?.let { iconBmps[id] = it }
            } catch (e: Exception) { }
        }
        for (a in appList) {                                     // icones oficiais em assets/icon_<id>.png substituem os embutidos
            try { act.assets.open("icon_" + a.id + ".png").use { st -> BitmapFactory.decodeStream(st)?.let { iconBmps[a.id] = it } } } catch (e: Exception) { }
        }
    }

    // ---------- estado ----------
    private var text = ""
    private var k1 = .22f; private var k2 = .08f; private var ipd = .064f
    private val head = floatArrayOf(0f, 0f, 0f)
    private var F = floatArrayOf(0f, 0f, -1f); private var R = floatArrayOf(1f, 0f, 0f); private var U = floatArrayOf(0f, 1f, 0f)
    private var hoverKey = ""
    private val dirtyIds = HashSet<String>()                    // janelas cujo bitmap precisa ser redesenhado
    private fun ui(vararg ids: String) { for (i in ids) dirtyIds.add(i) }
    private val levels = floatArrayOf(.45f, .55f, .65f, .75f, .85f)   // escala de renderizacao (adaptativa)
    private var level = 2
    private var rebuild = false
    private var frames = 0; private var acc = 0f; private var cool = 3f
    private var lastPoll = 0L; private var lastClock = 0L
    private var lastOvKey = ""
    private var prevPtrs = 0
    private val texDims = HashMap<Int, Long>()
    @Volatile var depth: DepthNet? = null                      // profundidade por IA (oclusao)
    private var lastAiCam: Bitmap? = null
    private var depthTex = 0; private var lastDepthVer = -1; private var lastDepthStatus = ""
    private var curEye = 0
    private var wDepth = 0; private var wVp = 0; private var wCrop = 0; private var wOcc = 0
    @Suppress("DEPRECATION")
    private fun displayRotation(): Int = act.windowManager.defaultDisplay.rotation
    @Volatile private var tap = false

    // ---------- sensores ----------
    private val sm = act.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val rm = FloatArray(9)
    @Volatile private var gazeF = floatArrayOf(0f, 0f, -1f)
    @Volatile private var gazeU = floatArrayOf(0f, 1f, 0f)
    @Volatile private var needCenter = true
    @Volatile private var yawOff = 0f

    override fun onResume() {
        super.onResume()
        val s = sm.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR) ?: sm.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        sm.registerListener(this, s, SensorManager.SENSOR_DELAY_GAME)
    }
    override fun onPause() { sm.unregisterListener(this); super.onPause() }
    override fun onAccuracyChanged(s: Sensor?, a: Int) {}
    @Suppress("DEPRECATION")
    override fun onSensorChanged(e: SensorEvent) {
        SensorManager.getRotationMatrixFromVector(rm, e.values)
        val gx = -rm[2]; val gy = -rm[5]; val gz = -rm[8]       // olhar = -Z do aparelho (ENU)
        val sg = if (act.windowManager.defaultDisplay.rotation == Surface.ROTATION_270) -1f else 1f
        val ux = sg * rm[0]; val uy = sg * rm[3]; val uz = sg * rm[6]
        gazeF = floatArrayOf(gx, gz, -gy)                        // ENU -> GL (X=leste, Y=cima, Z=-norte)
        gazeU = floatArrayOf(ux, uz, -uy)
        if (needCenter) { yawOff = atan2(gx, gy); needCenter = false }
    }

    private fun updateHead() {
        val a = yawOff; val ca = cos(a); val sa = sin(a)
        fun rot(v: FloatArray) = floatArrayOf(v[0] * ca + v[2] * sa, v[1], -v[0] * sa + v[2] * ca)
        val f = norm(rot(gazeF))
        val r = norm(cross(f, rot(gazeU)))
        F = f; R = r; U = cross(r, f)
    }

    // ---------- colocacao de janelas ----------
    private fun place(w: Win, p: FloatArray) {
        for (i in 0..2) w.pos[i] = p[i]
        val nrm = norm(floatArrayOf(head[0] - p[0], head[1] - p[1], head[2] - p[2]))
        var r = cross(floatArrayOf(0f, 1f, 0f), nrm)
        if (dot(r, r) < 1e-4f) r = floatArrayOf(1f, 0f, 0f)
        r = norm(r)
        w.r = r; w.n = nrm; w.u = cross(nrm, r)
    }
    private fun spawn(w: Win) {
        val a = atan2(F[0], -F[2]) + w.sa * (PI.toFloat() / 180f)
        place(w, floatArrayOf(head[0] + sin(a) * w.sd, head[1] + w.sy, head[2] - cos(a) * w.sd))
    }
    private fun toggle(w: Win) { if (w.visible) w.visible = false else { w.visible = true; spawn(w) } }

    // ---------- UI desenhada em bitmaps ----------
    private val p = Paint(Paint.ANTI_ALIAS_FLAG)
    private val tp = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE }
    private val hp = Paint(Paint.ANTI_ALIAS_FLAG)

    /** Fundo de vidro escuro com degrade e borda fina, como os paineis do Quest. */
    private fun panelBg(c: Canvas, w: Int, h: Int, r: Float) {
        p.style = Paint.Style.FILL
        p.shader = LinearGradient(0f, 0f, 0f, h.toFloat(), 0xF2202540.toInt(), 0xF2121626.toInt(), Shader.TileMode.CLAMP)
        c.drawRoundRect(0f, 0f, w.toFloat(), h.toFloat(), r, r, p)
        p.shader = null
        p.style = Paint.Style.STROKE; p.strokeWidth = 3f; p.color = 0x33FFFFFF
        c.drawRoundRect(1.5f, 1.5f, w - 1.5f, h - 1.5f, r, r, p)
        p.style = Paint.Style.FILL
    }

    private fun ensure(w: Win): Bitmap = w.bmp ?: Bitmap.createBitmap(w.bw, w.bh, Bitmap.Config.ARGB_8888).also { w.bmp = it }

    private fun key(c: Canvas, w: Win, id: String, label: String, r: RectF, size: Float = 36f, lit: Boolean = false) {
        val full = w.id + ":" + id
        val hot = hoverKey == full
        val prs = pressedKey == full
        val top = if (prs) 0xFF34D399.toInt() else if (hot) 0xFFA386FF.toInt() else if (lit) 0xFF5B6CF0.toInt() else 0xFF363D63.toInt()
        val bot = if (prs) 0xFF059669.toInt() else if (hot) 0xFF6A4BEA.toInt() else if (lit) 0xFF3D4BC8.toInt() else 0xFF252B49.toInt()
        p.style = Paint.Style.FILL
        p.shader = LinearGradient(r.left, r.top, r.left, r.bottom, top, bot, Shader.TileMode.CLAMP)
        c.drawRoundRect(r, 22f, 22f, p)
        p.shader = null
        p.style = Paint.Style.STROKE; p.strokeWidth = 2f; p.color = if (hot || prs) 0x88FFFFFF.toInt() else 0x22FFFFFF
        c.drawRoundRect(r, 22f, 22f, p)
        p.style = Paint.Style.FILL
        tp.textAlign = Paint.Align.CENTER; tp.textSize = size; tp.color = Color.WHITE
        c.drawText(label, r.centerX(), r.centerY() + size * .35f, tp)
        w.btns.add(Btn(id, r))
    }

    private fun drawTitle(w: Win) {
        val b = w.tBmp ?: Bitmap.createBitmap(w.tbw, w.tbh, Bitmap.Config.ARGB_8888).also { w.tBmp = it }
        b.eraseColor(0); val c = Canvas(b); w.tbBtns.clear()
        val hovWin = hoverKey.startsWith(w.id + ":")
        val grab = gw === w && mode == 2
        // barra branca de arrastar (como no Quest): pince nela e mova para levar a janela
        val pw = (w.tbw * .22f).coerceIn(110f, 220f) * (if (grab) 1.12f else 1f)
        val ph = if (grab) 18f else 14f
        p.shader = null; p.style = Paint.Style.FILL
        p.color = if (grab) 0xFFFFFFFF.toInt() else if (hovWin) 0xF2FFFFFF.toInt() else 0xB3FFFFFF.toInt()
        c.drawRoundRect(RectF(w.tbw / 2f - pw / 2f, 30f - ph / 2f, w.tbw / 2f + pw / 2f, 30f + ph / 2f), ph / 2f, ph / 2f, p)
        listOf("size", "close").forEachIndexed { i, id ->
            val left = w.tbw - 118f + i * 58f
            val r = RectF(left, 6f, left + 48f, 54f)
            w.tbBtns.add(Btn(id, r))
            if (!hovWin && !grab) return@forEachIndexed                  // botoes so aparecem ao apontar a janela (tela limpa)
            val hot = hoverKey == w.id + ":" + id
            p.style = Paint.Style.FILL
            p.color = if (hot) (if (id == "close") 0xFFE5484D.toInt() else 0xFF7C5CFF.toInt()) else 0xB32B3150.toInt()
            c.drawCircle(r.centerX(), r.centerY(), 24f, p)
            val cx = r.centerX(); val cy = r.centerY(); val q = 9f
            p.style = Paint.Style.STROKE; p.strokeWidth = 4f; p.strokeCap = Paint.Cap.ROUND; p.color = Color.WHITE
            if (id == "close") { c.drawLine(cx - q, cy - q, cx + q, cy + q, p); c.drawLine(cx - q, cy + q, cx + q, cy - q, p) }
            else {
                c.drawLine(cx - q, cy + q, cx + q, cy - q, p)
                c.drawLine(cx + q, cy - q, cx + q - 8f, cy - q, p); c.drawLine(cx + q, cy - q, cx + q, cy - q + 8f, p)
                c.drawLine(cx - q, cy + q, cx - q + 8f, cy + q, p); c.drawLine(cx - q, cy + q, cx - q, cy + q - 8f, p)
            }
            p.style = Paint.Style.FILL
        }
        w.tDirty = false
    }

    private fun icon(c: Canvas, w: Win, id: String, label: String, col: Int, r: RectF, size: Float, on: Boolean = false, bmp: Bitmap? = null) {
        val hot = hoverKey == w.id + ":" + id
        val g = if (hot) 6f else 0f
        val rr = RectF(r.left - g, r.top - g, r.right + g, r.bottom + g)
        p.style = Paint.Style.FILL
        if (bmp != null) p.shader = LinearGradient(rr.left, rr.top, rr.left, rr.bottom, 0xFF2E3558.toInt(), 0xFF1B2038.toInt(), Shader.TileMode.CLAMP)
        else p.color = col
        c.drawRoundRect(rr, 24f, 24f, p)
        p.shader = null
        if (bmp != null) {
            val m = rr.width() * .14f
            c.drawBitmap(bmp, null, RectF(rr.left + m, rr.top + m, rr.right - m, rr.bottom - m), ip)
        }
        p.style = Paint.Style.STROKE; p.strokeWidth = if (hot) 4f else 2f; p.color = if (hot) 0xFFB9A6FF.toInt() else 0x33FFFFFF
        c.drawRoundRect(rr, 24f, 24f, p)
        p.style = Paint.Style.FILL
        if (bmp == null) {
            tp.textAlign = Paint.Align.CENTER; tp.textSize = size; tp.color = Color.WHITE
            c.drawText(label, r.centerX(), r.centerY() + size * .35f, tp)
        }
        if (on) { p.color = Color.WHITE; c.drawCircle(r.centerX(), r.bottom + 12f, 5f, p) }
        w.btns.add(Btn(id, r))
    }

    private fun drawDock() {
        val w = dock; val b = ensure(w); b.eraseColor(0); val c = Canvas(b); w.btns.clear()
        panelBg(c, 1280, 128, 64f)
        tp.textAlign = Paint.Align.LEFT; tp.textSize = 34f; tp.color = Color.WHITE
        c.drawText(java.text.SimpleDateFormat("HH:mm", Locale.getDefault()).format(java.util.Date()), 36f, 76f, tp)
        val items = ArrayList<Triple<String, String, Int>>()
        items.add(Triple("lib", "LIB", 0xFF4A5572.toInt()))
        for (a in appList.take(7)) items.add(Triple("app:" + a.id, a.glyph, a.color))
        items.add(Triple("kb", "ABC", 0xFF3B82F6.toInt()))
        items.add(Triple("cfg", "AJU", 0xFF4A5572.toInt()))
        items.add(Triple("rc", "CTR", 0xFF16A34A.toInt()))
        items.forEachIndexed { i, (id, l, col) ->
            val on = (id == "lib" && apps.visible) || (id == "kb" && kb.visible) || (id == "cfg" && cfg.visible)
            icon(c, w, id, l, col, RectF(190f + i * 98f, 14f, 270f + i * 98f, 94f), 28f, on, iconBmps[id.removePrefix("app:")])
        }
        w.dirty = true
    }

    private fun drawAddr() {
        val w = addr; val b = ensure(w); b.eraseColor(0); val c = Canvas(b); w.btns.clear()
        panelBg(c, 1760, 112, 56f)
        key(c, w, "back", "<", RectF(20f, 16f, 110f, 96f), 34f)
        p.style = Paint.Style.FILL; p.color = 0xFF151929.toInt(); c.drawRoundRect(130f, 16f, 1560f, 96f, 40f, 40f, p)
        w.btns.add(Btn("addrf", RectF(130f, 16f, 1560f, 96f)))
        tp.textAlign = Paint.Align.LEFT; tp.textSize = 34f; tp.color = if (text.isEmpty()) 0xFF9AA3BD.toInt() else Color.WHITE
        c.drawText(if (text.isEmpty()) "Toque aqui para digitar um endereco ou busca" else text.takeLast(60), 160f, 68f, tp)
        key(c, w, "go", "Ir", RectF(1580f, 16f, 1740f, 96f), 34f)
        w.dirty = true
    }

    private fun drawApps() {
        val w = apps; val b = ensure(w); b.eraseColor(0); val c = Canvas(b); w.btns.clear()
        panelBg(c, 1040, 720, 44f)
        p.style = Paint.Style.FILL; p.color = 0xFF151929.toInt(); c.drawRoundRect(40f, 28f, 560f, 88f, 30f, 30f, p)
        tp.textAlign = Paint.Align.LEFT; tp.textSize = 28f; tp.color = 0xFF9AA3BD.toInt(); c.drawText("Buscar apps", 76f, 68f, tp)
        p.color = 0xFF7C5CFF.toInt(); c.drawRoundRect(600f, 36f, 740f, 80f, 22f, 22f, p)
        tp.textAlign = Paint.Align.CENTER; tp.textSize = 24f; tp.color = Color.WHITE; c.drawText("Todos", 670f, 67f, tp)
        appList.forEachIndexed { i, a ->
            val x = 45f + (i % 3) * 330f; val y = 116f + (i / 3) * 196f
            val r = RectF(x, y, x + 300f, y + 176f)
            val hot = hoverKey == w.id + ":app:" + a.id
            val ib = iconBmps[a.id]
            p.style = Paint.Style.FILL
            p.shader = LinearGradient(r.left, r.top, r.left, r.bottom,
                if (hot) 0xFF3A4270.toInt() else 0xFF2A3050.toInt(), if (hot) 0xFF272E55.toInt() else 0xFF1B2038.toInt(), Shader.TileMode.CLAMP)
            c.drawRoundRect(r, 30f, 30f, p)
            p.shader = null
            if (hot) { p.style = Paint.Style.STROKE; p.strokeWidth = 4f; p.color = 0xFF9D86FF.toInt(); c.drawRoundRect(r, 30f, 30f, p); p.style = Paint.Style.FILL }
            if (ib != null) {
                c.drawBitmap(ib, null, RectF(r.centerX() - 46f, r.top + 22f, r.centerX() + 46f, r.top + 114f), ip)
            } else {
                p.color = a.color; c.drawCircle(r.centerX(), r.top + 68f, 46f, p)
                tp.textAlign = Paint.Align.CENTER; tp.textSize = 34f; tp.color = Color.WHITE; c.drawText(a.glyph, r.centerX(), r.top + 80f, tp)
            }
            tp.textAlign = Paint.Align.CENTER; tp.textSize = 28f; tp.color = Color.WHITE
            c.drawText(a.name, r.centerX(), r.bottom - 26f, tp)
            w.btns.add(Btn("app:" + a.id, r))
        }
        w.dirty = true
    }

    private fun drawKb() {
        val w = kb; val b = ensure(w); b.eraseColor(0); val c = Canvas(b); w.btns.clear()
        panelBg(c, 1280, 624, 44f)
        // barra de previa: mostra o que esta sendo digitado e para onde vai (pagina ou endereco)
        p.style = Paint.Style.FILL; p.color = 0xFF151929.toInt(); c.drawRoundRect(40f, 14f, 1240f, 94f, 34f, 34f, p)
        tp.textAlign = Paint.Align.LEFT; tp.textSize = 28f; tp.color = 0xFF9D86FF.toInt()
        c.drawText(if (kbPage) "PAGINA" else "ENDERECO", 66f, 65f, tp)
        tp.textSize = 34f; tp.color = Color.WHITE
        c.drawText((if (kbPage) pageBuf else text).takeLast(44) + "|", 290f, 66f, tp)
        val y0 = 112f; val dy = 96f
        fun lab(ch: Char): String = if (shift && !sym && ch.isLetter()) ch.toString().uppercase() else ch.toString()
        val r1 = if (sym) "@#\$%&*-+()" else "qwertyuiop"
        val r2 = if (sym) "!?\"':;/_=" else "asdfghjkl"
        val r3 = if (sym) "[]{}<>|" else "zxcvbnm"
        "1234567890".forEachIndexed { i, ch -> key(c, w, "k:$ch", "$ch", RectF(40f + i * 120f, y0, 152f + i * 120f, y0 + 84f)) }
        r1.forEachIndexed { i, ch -> key(c, w, "k:$ch", lab(ch), RectF(40f + i * 120f, y0 + dy, 152f + i * 120f, y0 + dy + 84f)) }
        r2.forEachIndexed { i, ch -> key(c, w, "k:$ch", lab(ch), RectF(100f + i * 120f, y0 + 2 * dy, 212f + i * 120f, y0 + 2 * dy + 84f)) }
        val y3 = y0 + 3 * dy
        if (sym) key(c, w, "k:~", "~", RectF(40f, y3, 208f, y3 + 84f))
        else key(c, w, "k:shift", "Shift", RectF(40f, y3, 208f, y3 + 84f), 30f, shift)
        r3.forEachIndexed { i, ch -> key(c, w, "k:$ch", lab(ch), RectF(220f + i * 120f, y3, 332f + i * 120f, y3 + 84f)) }
        key(c, w, "k:del", "Apagar", RectF(1072f, y3, 1240f, y3 + 84f), 28f)
        val y4 = y0 + 4 * dy
        key(c, w, "k:sym", if (sym) "ABC" else "?123", RectF(40f, y4, 210f, y4 + 84f), 30f, sym)
        key(c, w, "k:,", ",", RectF(222f, y4, 334f, y4 + 84f))
        key(c, w, "k:sp", "Espaco", RectF(346f, y4, 946f, y4 + 84f), 32f)
        key(c, w, "k:.", ".", RectF(958f, y4, 1070f, y4 + 84f))
        key(c, w, "k:go", if (kbPage) "Enter" else "Ir", RectF(1082f, y4, 1240f, y4 + 84f), 30f)
        w.dirty = true
    }

    private fun drawCfg() {
        val w = cfg; val b = ensure(w); b.eraseColor(0); val c = Canvas(b); w.btns.clear()
        panelBg(c, 1000, 800, 44f)
        val rows = listOf(
            Triple("Lente k1", String.format(Locale.US, "%.2f", k1), "k1"),
            Triple("Lente k2", String.format(Locale.US, "%.2f", k2), "k2"),
            Triple("Olhos (IPD)", "${(ipd * 1000).roundToInt()} mm", "ipd"))
        rows.forEachIndexed { i, (l, v, id) ->
            val y = 30f + i * 100f
            tp.textAlign = Paint.Align.LEFT; tp.textSize = 34f; tp.color = Color.WHITE; c.drawText(l, 40f, y + 62f, tp)
            tp.textAlign = Paint.Align.CENTER; c.drawText(v, 540f, y + 62f, tp)
            key(c, w, "$id-", "-", RectF(640f, y, 740f, y + 84f)); key(c, w, "$id+", "+", RectF(770f, y, 870f, y + 84f))
        }
        key(c, w, "rc", "Recentrar", RectF(40f, 360f, 340f, 444f), 30f)
        key(c, w, "mode", if (touchMode) "Cursor: Dedo" else "Cursor: Raio", RectF(360f, 360f, 660f, 444f), 28f)
        key(c, w, "exit", "Sair", RectF(680f, 360f, 960f, 444f), 30f)
        key(c, w, "hand", if (dominantRight) "Mao dominante: Direita" else "Mao dominante: Esquerda", RectF(40f, 460f, 660f, 525f), 28f)
        key(c, w, "diag", if (showDiag) "Diagnostico: ON" else "Diagnostico: OFF", RectF(680f, 460f, 960f, 525f), 24f)
        tp.textAlign = Paint.Align.LEFT; tp.textSize = 25f; tp.color = 0xFFCBD2E8.toInt()
        c.drawText("Clique = pinca (polegar + indicador). Pince e arraste para rolar.", 40f, 580f, tp)
        val dnc = depth
        key(c, w, "occ", if (dnc == null) "Oclusao (IA): indisponivel" else if (dnc.enabled) "Oclusao (IA): Ligada" else "Oclusao (IA): Desligada",
            RectF(40f, 630f, 600f, 700f), 26f)
        tp.textAlign = Paint.Align.LEFT; tp.textSize = 24f; tp.color = 0xFF9AA3BD.toInt()
        c.drawText("IA: " + (dnc?.status ?: "-"), 630f, 675f, tp)
        tp.textAlign = Paint.Align.LEFT; tp.textSize = 26f; tp.color = 0xFF9AA3BD.toInt()
        c.drawText("As janelas ficam fixas no mundo virtual.", 40f, 760f, tp)
        w.dirty = true
    }

    // ---------- digitacao ----------
    private var kbPage = false                  // true: teclas vao para o campo da PAGINA; false: para a barra de endereco
    private var shift = false; private var sym = false
    private var pageBuf = ""                    // eco do que foi digitado na pagina (so para mostrar no teclado)
    private var pressedKey = ""; private var pressedT = 0f
    @Volatile private var edState = -1          // resultado da checagem "tem campo de texto focado?" (vem da thread de UI)

    private fun typeChar(ch: String) {
        if (kbPage) { pageBuf = (pageBuf + ch).takeLast(60); post { browser.typeText(ch) } } else text += ch
        shift = false
    }
    private fun typeDel() {
        if (kbPage) { pageBuf = pageBuf.dropLast(1); post { browser.backspace() } } else text = text.dropLast(1)
    }
    private fun typeEnter() {
        if (kbPage) { pageBuf = ""; post { browser.enter() }; kb.visible = false; kbPage = false } else go()
    }
    /** Depois de tocar na pagina, ve se caiu num campo de texto: se sim, abre o teclado ja apontando para a pagina. */
    private fun checkEditableSoon() {
        postDelayed({ browser.queryEditable { ed -> edState = if (ed) 1 else 0 } }, 300)
    }
    private fun applyEditable(ed: Boolean) {
        if (ed) { kbPage = true; pageBuf = ""; if (!kb.visible) { kb.visible = true; spawn(kb) }; ui("kb") }
        else if (kbPage && kb.visible) { kb.visible = false; kbPage = false; ui("kb") }
    }

    // ---------- acoes ----------
    private fun load(u: String) { browserWin.visible = true; post { browser.load(u) } }
    private fun go() {
        val t = text.trim(); if (t.isEmpty()) return
        val u = when {
            t.startsWith("http") -> t
            t.contains(' ') || !t.contains('.') -> "https://www.google.com/search?q=" + Uri.encode(t)
            else -> "https://$t"
        }
        load(u); kb.visible = false
    }
    private fun onButton(id: String) {
        if (id.startsWith("k:")) { pressedKey = "kb:" + id; pressedT = .14f }
        post { playSoundEffect(SoundEffectConstants.CLICK) }
        when {
            id == "addrf" -> { kbPage = false; if (!kb.visible) { kb.visible = true; spawn(kb) } }
            id == "back" -> post { browser.back() }
            id == "lib" -> toggle(apps)
            id == "go" -> go()
            id.startsWith("app:") -> appList.firstOrNull { it.id == id.substring(4) }?.let { load(it.url) }
            id == "kb" -> toggle(kb)
            id == "cfg" -> toggle(cfg)
            id == "rc" -> needCenter = true
            id == "mode" -> touchMode = !touchMode
            id == "hand" -> dominantRight = !dominantRight
            id == "diag" -> showDiag = !showDiag
            id == "occ" -> depth?.let { it.enabled = !it.enabled }
            id == "exit" -> post { onExit() }
            id == "k1-" -> k1 = (k1 - .03f).coerceAtLeast(0f)
            id == "k1+" -> k1 = (k1 + .03f).coerceAtMost(.6f)
            id == "k2-" -> k2 = (k2 - .03f).coerceAtLeast(0f)
            id == "k2+" -> k2 = (k2 + .03f).coerceAtMost(.4f)
            id == "ipd-" -> ipd = (ipd - .001f).coerceAtLeast(.05f)
            id == "ipd+" -> ipd = (ipd + .001f).coerceAtMost(.075f)
            id == "k:del" -> typeDel()
            id == "k:sp" -> typeChar(" ")
            id == "k:go" -> typeEnter()
            id == "k:shift" -> shift = !shift
            id == "k:sym" -> { sym = !sym; shift = false }
            id.startsWith("k:") -> typeChar(if (shift) id.substring(2).uppercase() else id.substring(2))
        }
        ui("dock", "addr", "cfg", "kb")
    }

    private fun sendTouch(action: Int, x: Float, y: Float) { post { browser.touch(action, x, y) } }

    // ---------- interacao: raio (mao ou olhar) contra janelas ----------
    private class Hit(val win: Win, val title: Boolean, val bx: Float, val by: Float, val t: Float, val lx: Float, val ly: Float)
    private class Ptr(val hand: Hand, val ox: Float, val oy: Float, val tx: Float, val ty: Float, val pinch: Boolean, val k: Int, val palm: Float)

    private val ptrs = ArrayList<Ptr>()
    private class MBtn(val id: String, val cx: Float, val cy: Float, val r: Float)
    private val menuBtns = ArrayList<MBtn>()                        // botoes redondos nas pontas dos dedos
    private val menuIds = arrayOf("lib", "kb", "rc", "cfg")        // indicador, medio, anelar, minimo
    private var hostP: Ptr? = null; private var pointerP: Ptr? = null
    private var menuHover = ""; private var menuPressing = false; private var prevMenuPress = false
    private var mDwell = 0f; private var mDwellId = ""
    private var dominantRight = true                                // mao dominante (a outra abre o menu)
    private var gPalm0 = 1f
    private var gDist0 = 1f                                         // distancia inicial entre as maos (redimensionar)
    private val prevMid = BooleanArray(2); private val prevRing = BooleanArray(2); private val prevPinky = BooleanArray(2)
    private var prevHostPinch = false; private var menuShownPrev = false
    private val sx = FloatArray(2); private val sy = FloatArray(2); private val sOn = BooleanArray(2)
    private var showDiag = false                                    // texto de diagnostico no topo (desligado = tela limpa)
    private val freeX = FloatArray(2); private val freeY = FloatArray(2)      // ponteiro de ANTES da pinca (modo dedo)
    private val stX = FloatArray(2); private val stY = FloatArray(2)          // ponto estavel (juntas dos dedos), nao muda ao pincar
    private val stX0 = FloatArray(2); private val stY0 = FloatArray(2); private val wasPress = BooleanArray(2)
    private val palmSm = FloatArray(2)
    private var touchMode = false                                  // false = RAIO estavel (padrao); true = cursor na ponta do dedo
    private var dnBx = 0f; private var dnBy = 0f; private var dragging = false
    private val TAP_SLOP = 22f                                     // px da janela: abaixo disso o clique nao vira arrasto (nao treme)
    private var cropL = 0f; private var cropT = 0f; private var cropS = 1f
    private var cu0 = 0f; private var cv0 = 0f; private var cus = 1f; private var cvs = 1f

    private var mode = 0                    // 0 nada, 1 toque no navegador, 2 mover janela, 3 redimensionar
    private var gw: Win? = null
    private var gDist = 0f; private val gOff = FloatArray(3)
    private var gScale = 1f; private var gLx = 0f
    private var lastBx = 0f; private var lastBy = 0f
    private var pinching = false
    private var dwell = 0f; private var dwellId = ""
    private var rayO = floatArrayOf(0f, 0f, 0f); private var rayD = floatArrayOf(0f, 0f, -1f)
    private var curHit: Hit? = null

    private fun rayFromEye(px: Float, py: Float): FloatArray {
        val ew = fw / 2f; val eh = fh.toFloat(); val asp = ew / eh
        val dx = px / (ew / 2f) * asp; val dy = -py / (eh / 2f)
        return norm(floatArrayOf(R[0] * dx + U[0] * dy + F[0], R[1] * dx + U[1] * dy + F[1], R[2] * dx + U[2] * dy + F[2]))
    }

    private fun planeHit(w: Win, o: FloatArray, d: FloatArray): Hit? {
        val den = dot(d, w.n); if (abs(den) < 1e-4f) return null
        val base = (w.pos[0] - o[0]) * w.n[0] + (w.pos[1] - o[1]) * w.n[1] + (w.pos[2] - o[2]) * w.n[2]
        var t = base / den
        if (t < .1f) return null
        var rel = floatArrayOf(o[0] + d[0] * t - w.pos[0], o[1] + d[1] * t - w.pos[1], o[2] + d[2] * t - w.pos[2])
        var lx = dot(rel, w.r)
        t = (base + lx * lx / (2f * CURVE_R)) / den              // corrige para a superficie curva
        if (t < .1f) return null
        rel = floatArrayOf(o[0] + d[0] * t - w.pos[0], o[1] + d[1] * t - w.pos[1], o[2] + d[2] * t - w.pos[2])
        lx = dot(rel, w.r); val ly = dot(rel, w.u)
        val ws = w.w * w.scale; val hs = w.h * w.scale
        return Hit(w, false, ((lx / ws + .5f) * w.bw).coerceIn(0f, w.bw.toFloat()), ((.5f - ly / hs) * w.bh).coerceIn(0f, w.bh.toFloat()), t, lx, ly)
    }

    private fun pick(o: FloatArray, d: FloatArray): Hit? {
        var best: Hit? = null
        for (w in wins) {
            if (!w.visible) continue
            val ph = planeHit(w, o, d) ?: continue
            val b = best
            if (b != null && ph.t >= b.t) continue
            val ws = w.w * w.scale; val hs = w.h * w.scale; val th = w.thM
            if (abs(ph.lx) > ws / 2) continue
            if (abs(ph.ly) <= hs / 2) best = ph
            else if (w.hasTitle && ph.ly < -hs / 2 && ph.ly >= -hs / 2 - th * 2f)       // zona de agarrar generosa (2x a barra)
                best = Hit(w, true, (ph.lx / ws + .5f) * w.tbw, (((-hs / 2 - ph.ly) / th) * w.tbh).coerceAtMost(w.tbh.toFloat()), ph.t, ph.lx, ph.ly)
        }
        return best
    }

    /** Botao sob o ponteiro, com uma folga em volta (imã) e escolhendo o mais proximo do centro. */
    private fun btnAt(h: Hit): String? {
        val list = if (h.title) h.win.tbBtns else h.win.btns
        val m = if (h.title) 12f else 20f
        var best: String? = null; var bd = Float.MAX_VALUE
        for (b in list) {
            val r = b.r
            if (h.bx >= r.left - m && h.bx <= r.right + m && h.by >= r.top - m && h.by <= r.bottom + m) {
                val dd = hypot(h.bx - r.centerX(), h.by - r.centerY())
                if (dd < bd) { bd = dd; best = b.id }
            }
        }
        return best
    }

    private fun activate(h: Hit, allowTouch: Boolean) {
        val id = btnAt(h)
        if (h.title) { if (id == "close" || id == "min") h.win.visible = false }
        else if (h.win === browserWin) { if (allowTouch) { sendTouch(MotionEvent.ACTION_DOWN, h.bx, h.by); sendTouch(MotionEvent.ACTION_UP, h.bx, h.by); checkEditableSoon() } }
        else if (id != null) onButton(id)
    }

    private fun interact(dt: Float) {
        // ponteiros das maos (em pixels do olho, antes da lente)
        ptrs.clear()
        val cb = curFr?.bmp; val hs = curFr?.hands ?: emptyList()
        val ew = fw / 2f; val eh = fh.toFloat()
        val seen = BooleanArray(2)
        if (cb != null) for (hd in hs) {
            if (hd.conf < .12f) continue                             // so ignora quando a mao ja sumiu de vez
            val k = hd.slot.coerceIn(0, 1)
            fun X(i: Int) = (hd.pts[2 * i] * cb.width - cropL) * cropS - ew / 2f
            fun Y(i: Int) = (hd.pts[2 * i + 1] * cb.height - cropT) * cropS - eh / 2f
            val ox = (X(4) + X(8)) / 2f; val oy = (Y(4) + Y(8)) / 2f          // ponto da pinca (entre polegar e indicador)
            val palm = hypot(X(5) - X(17), Y(5) - Y(17)).coerceAtLeast(5f)    // largura dos nos dos dedos
            // ponto ESTAVEL: meio entre as juntas do indicador e do medio; nao se mexe quando os dedos fecham
            val kx = (X(5) + X(9)) / 2f; val ky = (Y(5) + Y(9)) / 2f
            val ddx = kx - X(0); val ddy = ky - Y(0)
            val rawX = if (touchMode) X(8) else kx + ddx * 1.05f
            val rawY = if (touchMode) Y(8) else ky + ddy * 1.05f
            if (!sOn[k]) {
                sx[k] = rawX; sy[k] = rawY; stX[k] = kx; stY[k] = ky; freeX[k] = rawX; freeY[k] = rawY
                palmSm[k] = palm; wasPress[k] = false
            } else {
                // suavizacao por tempo: parado = firme; mexendo rapido = quase sem atraso
                val dd = hypot(rawX - sx[k], rawY - sy[k])
                val rate = (10f + 110f * dd / palm).coerceIn(10f, 60f)
                val al = 1f - exp(-rate * dt)
                sx[k] += (rawX - sx[k]) * al; sy[k] += (rawY - sy[k]) * al
                stX[k] += (kx - stX[k]) * al; stY[k] += (ky - stY[k]) * al
                palmSm[k] += (palm - palmSm[k]) * (1f - exp(-6f * dt))
            }
            sOn[k] = true; seen[k] = true
            val press = hd.pinch
            if (!press && hd.strength < .15f) { freeX[k] = sx[k]; freeY[k] = sy[k] }
            if (press && !wasPress[k]) { wasPress[k] = true; stX0[k] = stX[k]; stY0[k] = stY[k] }
            else if (!press) wasPress[k] = false
            // modo dedo: a pinca desloca a ponta do dedo; entao o cursor fica onde estava ANTES e so segue o movimento da mao
            val anchored = press && wasPress[k] && touchMode
            val px = if (anchored) freeX[k] + (stX[k] - stX0[k]) else sx[k]
            val py = if (anchored) freeY[k] + (stY[k] - stY0[k]) else sy[k]
            ptrs.add(Ptr(hd, ox, oy, px, py, press, k, palmSm[k]))
        }
        for (k in 0..1) if (!seen[k]) sOn[k] = false
        // a mao nao dominante abre o menu nas pontas dos dedos; a dominante aponta e clica
        var host: Ptr? = null; var pointer: Ptr? = null
        if (ptrs.size >= 2) {
            val sorted = ptrs.sortedBy { it.hand.pts[0] }
            if (dominantRight) { host = sorted[0]; pointer = sorted[1] } else { host = sorted[1]; pointer = sorted[0] }
        }
        else if (ptrs.size == 1) { host = ptrs[0]; pointer = ptrs[0] }
        hostP = host; pointerP = pointer
        val act = pointer

        menuBtns.clear()
        if (host != null && cb != null && ptrs.size >= 2) {          // menu nos dedos so com as duas maos (uma mao aberta so aponta e clica)
            val hh = host.hand
            fun HX(i: Int) = (hh.pts[2 * i] * cb.width - cropL) * cropS - ew / 2f
            fun HY(i: Int) = (hh.pts[2 * i + 1] * cb.height - cropT) * cropS - eh / 2f
            var ext = 0                                              // dedos estendidos (mao aberta)
            for (j in 0..3) {
                val mcp = 5 + 4 * j; val tip = 8 + 4 * j
                if (hypot(HX(tip) - HX(0), HY(tip) - HY(0)) > hypot(HX(mcp) - HX(0), HY(mcp) - HY(0)) * 1.6f) ext++
            }
            if (ext >= 3 && !hh.pinch) {
                val pl = hypot(HX(5) - HX(17), HY(5) - HY(17)).coerceAtLeast(20f)
                val r = (pl * .15f).coerceIn(16f, 34f)
                for (j in 0..3) {
                    val tip = 8 + 4 * j; val dip = 7 + 4 * j
                    var ux = HX(tip) - HX(dip); var uy = HY(tip) - HY(dip)
                    val nn = hypot(ux, uy).coerceAtLeast(1f); ux /= nn; uy /= nn
                    menuBtns.add(MBtn(menuIds[j], HX(tip) + ux * r * 1.5f, HY(tip) + uy * r * 1.5f, r))
                }
            }
        }
        val pressAllowed = !(menuBtns.isNotEmpty() && pointer === host)
        var mId: String? = null; var mHand: Ptr? = null
        if (mode == 0 && !pinching && menuBtns.isNotEmpty()) {
            if (pointer != null && pointer !== host) {
                for (mb in menuBtns) if (hypot(pointer.tx - mb.cx, pointer.ty - mb.cy) < mb.r * 1.25f) { mId = mb.id; mHand = pointer }
            }
            if (mId == null) for (mb in menuBtns) if (hypot(mb.cx, mb.cy) < mb.r * 1.25f) mId = mb.id    // olhar
        }
        menuHover = mId ?: ""
        val menuPress = mHand?.pinch == true
        menuPressing = menuPress
        if (mId != null) {
            if (mHand != null) { if (menuPress && !prevMenuPress) onButton(mId); mDwell = 0f }
            else { if (mId == mDwellId) mDwell += dt else mDwell = 0f; mDwellId = mId; if (mDwell >= 1.2f) { onButton(mId); mDwell = -.8f } }
        } else { mDwell = 0f; mDwellId = "" }
        prevMenuPress = menuPress

        // gestos extras (mapeados por mim): pinca do dedo medio = voltar, anelar = recentrar, minimo = teclado
        for (q in ptrs) {
            val kk = q.k
            if (q.hand.mid && !prevMid[kk]) onButton("back")
            if (q.hand.ring && !prevRing[kk]) onButton("rc")
            if (q.hand.pinky && !prevPinky[kk]) onButton("kb")
            prevMid[kk] = q.hand.mid; prevRing[kk] = q.hand.ring; prevPinky[kk] = q.hand.pinky
        }
        for (kk in 0..1) if (!seen[kk]) { prevMid[kk] = false; prevRing[kk] = false; prevPinky[kk] = false }
        // gesto de sistema: pinca da mao do menu (com a outra mao livre) abre/fecha a biblioteca, como o botao Meta
        val hostPinch = host?.hand?.pinch == true
        if (ptrs.size >= 2 && hostPinch && !prevHostPinch && menuShownPrev && pointer?.hand?.pinch != true) { toggle(apps); ui("dock") }
        prevHostPinch = hostPinch
        menuShownPrev = menuBtns.isNotEmpty()

        val o = floatArrayOf(head[0], head[1], head[2])
        val d = if (act != null) rayFromEye(act.tx, act.ty) else F
        rayO = o; rayD = d

        // duas maos: pinca nas duas sobre a mesma janela = redimensionar (afastar = aumentar)
        val bothPinch = ptrs.size >= 2 && ptrs[0].hand.pinch && ptrs[1].hand.pinch
        if (bothPinch) {
            val pa = ptrs[0]; val pb = ptrs[1]
            val dist2 = hypot(pa.tx - pb.tx, pa.ty - pb.ty).coerceAtLeast(10f)
            if (mode == 5) {
                val w5 = gw
                if (w5 != null) w5.scale = (gScale * dist2 / gDist0).coerceIn(.4f, 3f)
                curHit = null
                return
            } else if (mode == 0 && !pinching) {
                val ha = pick(o, rayFromEye(pa.tx, pa.ty)); val hb = pick(o, rayFromEye(pb.tx, pb.ty))
                if (ha != null && hb != null && ha.win === hb.win && !ha.title && !hb.title) {
                    mode = 5; gw = ha.win; gScale = ha.win.scale; gDist0 = dist2
                    curHit = null
                    return
                }
            }
        } else if (mode == 5) {
            mode = 0; gw = null
            pinching = ptrs.any { it.hand.pinch }
        }

        val h = if (mId != null) null else if (mode == 0) pick(o, d) else gw?.let { planeHit(it, o, d) }
        curHit = h
        val id = if (mode == 0 && h != null) btnAt(h) else null
        val key = if (mode == 0 && h != null) h.win.id + ":" + (id ?: "") else ""
        if (key != hoverKey) { ui(hoverKey.substringBefore(':'), key.substringBefore(':')); hoverKey = key }

        if (tap) { tap = false; if (h != null && mode == 0) activate(h, true) }

        val pinch = act?.pinch == true && pressAllowed && mId == null
        if (pinch && !pinching) {
            pinching = true; dwell = -.8f
            if (h != null) {
                if (h.title) {
                    when (id) {
                        "close", "min" -> h.win.visible = false
                        "size" -> { mode = 3; gw = h.win; gScale = h.win.scale; gLx = h.lx }
                        else -> {
                            mode = 2; gw = h.win; gDist = h.t; gPalm0 = act?.palm ?: 1f
                            ui(h.win.id)
                            for (i in 0..2) gOff[i] = h.win.pos[i] - (o[i] + d[i] * h.t)
                        }
                    }
                } else if (h.win === browserWin) {
                    mode = 1; gw = h.win; lastBx = h.bx; lastBy = h.by; dnBx = h.bx; dnBy = h.by; dragging = false
                    sendTouch(MotionEvent.ACTION_DOWN, h.bx, h.by)
                } else if (id != null) onButton(id)
            }
        } else if (pinch) {
            val w = gw
            if (w != null && h != null) when (mode) {
                1 -> {                                           // toque so vira arrasto depois de passar da folga (clique firme)
                    if (!dragging && hypot(h.bx - dnBx, h.by - dnBy) > TAP_SLOP) dragging = true
                    if (dragging) { lastBx = h.bx; lastBy = h.by; sendTouch(MotionEvent.ACTION_MOVE, h.bx, h.by) }
                }
                2 -> {                                            // mover; levar a mao para perto/longe puxa/empurra a janela
                    val pn = (act?.palm ?: gPalm0).coerceAtLeast(1f)
                    val lf = ln(gPalm0 / pn)
                    val aa = abs(lf)
                    val ef = if (aa < .08f) 0f else (aa - .08f) * sign(lf)       // zona morta: tremor do tamanho da mao nao mexe a janela
                    val gd = (gDist * exp(ef * 1.4f)).coerceIn(.8f, 8f)
                    val al = 1f - exp(-20f * dt)                                  // janela acompanha suave, sem tremer
                    val tgt = floatArrayOf(o[0] + d[0] * gd + gOff[0], o[1] + d[1] * gd + gOff[1], o[2] + d[2] * gd + gOff[2])
                    place(w, floatArrayOf(w.pos[0] + (tgt[0] - w.pos[0]) * al, w.pos[1] + (tgt[1] - w.pos[1]) * al, w.pos[2] + (tgt[2] - w.pos[2]) * al))
                }
                3 -> { w.scale = (gScale * (1f + (h.lx - gLx) / (w.w * gScale))).coerceIn(.4f, 3f) }
            }
        } else if (pinching) {
            pinching = false
            if (mode == 1) { sendTouch(MotionEvent.ACTION_UP, lastBx, lastBy); if (!dragging) checkEditableSoon() }
            val g0 = gw; if (mode == 2 && g0 != null) ui(g0.id)
            mode = 0; gw = null
        }

        if (!pinching && ptrs.isEmpty()) {          // sem maos: olhar fixo em botao = clique
            if (id != null && id == dwellId) dwell += dt else dwell = 0f
            dwellId = id ?: ""
            if (dwell >= 1.2f && h != null) { activate(h, false); dwell = -.8f }
        } else if (ptrs.isNotEmpty()) dwell = 0f
    }

    // Toque na tela (se o VR Box tiver botao) = clique no ponto do olhar
    override fun onTouchEvent(e: MotionEvent): Boolean {
        if (e.action == MotionEvent.ACTION_DOWN) tap = true
        return true
    }

    // ---------- overlay 2D (maos, laser, diagnostico) ----------
    private var ovBmp: Bitmap? = null
    private var ovTex = 0
    // ---------- mao estilo Quest: vidro translucido + contorno branco fino ----------
    private val FING = arrayOf(intArrayOf(1, 2, 3, 4), intArrayOf(5, 6, 7, 8), intArrayOf(9, 10, 11, 12), intArrayOf(13, 14, 15, 16), intArrayOf(17, 18, 19, 20))
    private val FR = floatArrayOf(.150f, .128f, .134f, .124f, .108f)      // meia-largura da base de cada dedo (x largura da palma)
    private val TAPER = floatArrayOf(1f, .93f, .86f, .78f)                // afinamento ate a ponta
    private val hpath = Path()
    private val hpt = Paint(Paint.ANTI_ALIAS_FLAG)
    private val clearX = PorterDuffXfermode(PorterDuff.Mode.CLEAR)
    private val qx = FloatArray(8); private val qy = FloatArray(8)

    /** Disco como poligono (sentido anti-horario na tela, igual ao das capsulas: a uniao nao fura). */
    private fun addDisc(pa: Path, cx: Float, cy: Float, r: Float) {
        val n = 22
        for (i in 0 until n) {
            val a = 2f * PI.toFloat() * i / n
            val x = cx + r * cos(a); val y = cy - r * sin(a)
            if (i == 0) pa.moveTo(x, y) else pa.lineTo(x, y)
        }
        pa.close()
    }

    /** Capsula afinada entre dois pontos (dois discos + trapezio). */
    private fun addCap(pa: Path, ax: Float, ay: Float, ar: Float, bx: Float, by2: Float, br: Float) {
        addDisc(pa, ax, ay, ar); addDisc(pa, bx, by2, br)
        val dx = bx - ax; val dy = by2 - ay
        val l = hypot(dx, dy)
        if (l < .5f) return
        val nx = -dy / l; val ny = dx / l
        pa.moveTo(ax + nx * ar, ay + ny * ar); pa.lineTo(bx + nx * br, by2 + ny * br)
        pa.lineTo(bx - nx * br, by2 - ny * br); pa.lineTo(ax - nx * ar, ay - ny * ar); pa.close()
    }

    /** Monta a silhueta inteira da mao (dedos afinando + palma) como UM caminho; grow engorda tudo (para o contorno). */
    private fun buildHand(pa: Path, xs: FloatArray, ys: FloatArray, span: Float, grow: Float) {
        pa.reset(); pa.fillType = Path.FillType.WINDING
        val pts = FloatArray(8)
        for (f in 0 until 5) {
            val ch = FING[f]
            val r0 = span * FR[f]
            for (j in 0..3) { pts[2 * j] = xs[ch[j]]; pts[2 * j + 1] = ys[ch[j]] }
            val dx = pts[6] - pts[4]; val dy = pts[7] - pts[5]
            val l = hypot(dx, dy).coerceAtLeast(1e-3f)
            val pb = min(r0 * TAPER[3] * .8f, l * .9f)                   // o ponto da ponta e a extremidade da polpa, nao o centro
            pts[6] -= dx / l * pb; pts[7] -= dy / l * pb
            for (j in 0..2) addCap(pa, pts[2 * j], pts[2 * j + 1], r0 * TAPER[j] + grow, pts[2 * j + 2], pts[2 * j + 3], r0 * TAPER[j + 1] + grow)
        }
        // palma: poligono (pulso, mindinho..indicador, base do polegar) + capsulas nas bordas
        val wx = xs[0]; val wy = ys[0]
        var ux = xs[9] - wx; var uy = ys[9] - wy
        val ul = hypot(ux, uy).coerceAtLeast(1e-3f); ux /= ul; uy /= ul
        var nx = -uy; var ny = ux
        if (nx * (xs[17] - wx) + ny * (ys[17] - wy) < 0f) { nx = -nx; ny = -ny }       // n aponta para o lado do mindinho
        val hw = span * .30f
        qx[0] = wx + nx * hw - ux * span * .04f; qy[0] = wy + ny * hw - uy * span * .04f
        qx[1] = xs[17]; qy[1] = ys[17]; qx[2] = xs[13]; qy[2] = ys[13]; qx[3] = xs[9]; qy[3] = ys[9]
        qx[4] = xs[5]; qy[4] = ys[5]; qx[5] = xs[2]; qy[5] = ys[2]; qx[6] = xs[1]; qy[6] = ys[1]
        qx[7] = wx - nx * hw - ux * span * .04f; qy[7] = wy - ny * hw - uy * span * .04f
        var area = 0f
        for (i in 0 until 8) { val j = (i + 1) % 8; area += qx[i] * qy[j] - qx[j] * qy[i] }
        val rp = span * .13f + grow
        if (area > 0f) {                                                  // garante o mesmo sentido das capsulas
            for (i in 0 until 4) { var t = qx[i]; qx[i] = qx[7 - i]; qx[7 - i] = t; t = qy[i]; qy[i] = qy[7 - i]; qy[7 - i] = t }
        }
        pa.moveTo(qx[0], qy[0]); for (i in 1 until 8) pa.lineTo(qx[i], qy[i]); pa.close()
        for (i in 0 until 8) { val j = (i + 1) % 8; addCap(pa, qx[i], qy[i], rp, qx[j], qy[j], rp) }
    }

    private fun drawHand(c: Canvas, q: Ptr, cb: Bitmap, ew: Float, eh: Float) {
        val hd = q.hand
        val fade = (hd.conf * 2f).coerceIn(0f, 1f)
        if (fade <= .02f) return
        val xs = FloatArray(21); val ys = FloatArray(21)
        for (i in 0 until 21) {
            xs[i] = (hd.pts[2 * i] * cb.width - cropL) * cropS - ew / 2f
            ys[i] = (hd.pts[2 * i + 1] * cb.height - cropT) * cropS - eh / 2f
        }
        val span = hypot(xs[5] - xs[17], ys[5] - ys[17]).coerceAtLeast(24f)
        val ow = (span * .022f).coerceIn(1.6f, 3.4f)                      // espessura do contorno
        fun al(a: Int) = ((a * fade).toInt() shl 24)                      // alfa ja com o fade da perda de rastreio
        val pt = hpt
        pt.style = Paint.Style.FILL; pt.shader = null; pt.xfermode = null

        // 1 sombra suave deslocada (da profundidade, como a sombra da mao no chao do Quest)
        buildHand(hpath, xs, ys, span, ow * .5f)
        c.save(); c.translate(span * .03f, span * .05f)
        pt.color = al(58); c.drawPath(hpath, pt); c.restore()
        // 2 halo do contorno
        buildHand(hpath, xs, ys, span, ow * 5.5f)
        pt.color = al(34) or 0x00BFD6FF; c.drawPath(hpath, pt)
        // 3 contorno branco fino
        buildHand(hpath, xs, ys, span, ow)
        pt.color = al(235) or 0x00FFFFFF; c.drawPath(hpath, pt)
        // 4 apaga o miolo (so sobra o anel) e 5 enche com vidro translucido em degrade
        buildHand(hpath, xs, ys, span, 0f)
        pt.xfermode = clearX; pt.color = Color.BLACK; c.drawPath(hpath, pt); pt.xfermode = null
        pt.shader = LinearGradient(xs[12], ys[12], xs[0], ys[0], al(66) or 0x00EAF3FF, al(40) or 0x009FB6F2, Shader.TileMode.CLAMP)
        pt.color = Color.WHITE; c.drawPath(hpath, pt); pt.shader = null
        // brilho nas pontas dos dedos
        pt.color = al(46) or 0x00FFFFFF
        for (f in 0 until 5) { val t = FING[f][3]; c.drawCircle(xs[t], ys[t], span * FR[f] * .55f, pt) }

        // raio fino do punho ate o cursor (so no modo raio)
        val aiming = q === pointerP && !(menuBtns.isNotEmpty() && pointerP === hostP)
        if (aiming && !touchMode) {
            hp.style = Paint.Style.STROKE; hp.strokeWidth = 1.8f; hp.strokeCap = Paint.Cap.ROUND
            hp.shader = LinearGradient(q.ox, q.oy, q.tx, q.ty, al(0) or 0x00FFFFFF, al(150) or 0x00FFFFFF, Shader.TileMode.CLAMP)
            c.drawLine(q.ox, q.oy, q.tx, q.ty, hp); hp.shader = null
        }
        // indicador de pinca (como o ponto do Quest): anel que fecha ao aproximar, ponto brilhante ao pincar
        val prog = hd.strength
        if (aiming && (prog > .08f || q.pinch)) {
            val r0 = span * .26f
            if (q.pinch) {
                pt.shader = RadialGradient(q.ox, q.oy, span * .22f, al(240) or 0x00FFFFFF, al(0) or 0x00FFFFFF, Shader.TileMode.CLAMP)
                pt.color = Color.WHITE; c.drawCircle(q.ox, q.oy, span * .22f, pt); pt.shader = null
                pt.color = al(255) or 0x00FFFFFF; c.drawCircle(q.ox, q.oy, span * .05f, pt)
            } else {
                hp.shader = null; hp.style = Paint.Style.STROKE; hp.strokeWidth = 2.2f
                hp.color = al((70 + 150 * prog).toInt()) or 0x00FFFFFF
                c.drawCircle(q.ox, q.oy, r0 * (1.05f - .75f * prog), hp)
            }
        }
        pt.style = Paint.Style.FILL
    }

    private fun drawOverlay() {
        val b = ovBmp ?: return
        b.eraseColor(0)
        val c = Canvas(b)
        val ew = fw / 2f; val eh = fh.toFloat()
        c.translate(b.width / 2f, b.height / 2f); c.scale(b.width / ew, b.height / eh)
        if (showDiag) {
            tp.textAlign = Paint.Align.CENTER; tp.textSize = 22f; tp.color = 0xAAFFFFFF.toInt()
            c.drawText("camera: " + (if (cam != null) "ok" else "nao") + "   maos: " + hands.size + (if (touchMode) "   dedo" else "   raio") +
                (if (pinching) "   CLIQUE" else "") + "   Q" + (level + 1) +
                "   ia: " + (depth?.status ?: "-"), 0f, -eh / 2f + 50f, tp)
        }
        val cb = cam ?: return
        for (q in ptrs) drawHand(c, q, cb, ew, eh)
        for (mb in menuBtns) drawMenuBtn(c, mb)
        hp.style = Paint.Style.FILL
    }

    /** Botao redondo na ponta do dedo, com icone desenhado (casa, teclado, centrar, engrenagem). */
    private fun drawMenuBtn(c: Canvas, mb: MBtn) {
        val hot = menuHover == mb.id
        val x = mb.cx; val y = mb.cy; val r = mb.r
        hp.shader = null
        hp.style = Paint.Style.FILL
        hp.color = if (hot) (if (menuPressing) 0xFF22C55E.toInt() else 0xFF6A6A6A.toInt()) else 0xE62A2A2A.toInt()
        c.drawCircle(x, y, r, hp)
        hp.style = Paint.Style.STROKE; hp.strokeWidth = 2f; hp.color = 0xAAE6E6E6.toInt()
        c.drawCircle(x, y, r, hp)
        hp.color = Color.WHITE; hp.strokeWidth = max(2f, r * .09f)
        hp.strokeCap = Paint.Cap.ROUND; hp.strokeJoin = Paint.Join.ROUND
        when (mb.id) {
            "lib" -> {
                val roof = Path()
                roof.moveTo(x - r * .55f, y - r * .02f); roof.lineTo(x, y - r * .55f); roof.lineTo(x + r * .55f, y - r * .02f)
                c.drawPath(roof, hp)
                c.drawRect(x - r * .38f, y - r * .02f, x + r * .38f, y + r * .48f, hp)
            }
            "kb" -> {
                c.drawRoundRect(RectF(x - r * .6f, y - r * .32f, x + r * .6f, y + r * .32f), r * .1f, r * .1f, hp)
                hp.style = Paint.Style.FILL
                for (i in 0..2) for (j in 0..1) {
                    val kx = x - r * .42f + i * r * .34f; val ky = y - r * .18f + j * r * .22f
                    c.drawRect(kx, ky, kx + r * .16f, ky + r * .12f, hp)
                }
            }
            "rc" -> {
                c.drawCircle(x, y, r * .36f, hp)
                hp.style = Paint.Style.FILL; c.drawCircle(x, y, r * .1f, hp); hp.style = Paint.Style.STROKE
                c.drawLine(x, y - r * .46f, x, y - r * .66f, hp); c.drawLine(x, y + r * .46f, x, y + r * .66f, hp)
                c.drawLine(x - r * .46f, y, x - r * .66f, y, hp); c.drawLine(x + r * .46f, y, x + r * .66f, y, hp)
            }
            else -> {
                c.drawCircle(x, y, r * .3f, hp)
                for (i in 0..7) {
                    val a = i * (PI.toFloat() / 4f)
                    c.drawLine(x + cos(a) * r * .42f, y + sin(a) * r * .42f, x + cos(a) * r * .62f, y + sin(a) * r * .62f, hp)
                }
            }
        }
        hp.style = Paint.Style.FILL
    }

    private fun upDepth(d: ByteArray) {
        if (depthTex == 0) depthTex = GlUtil.tex() else GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, depthTex)
        val bb = ByteBuffer.allocateDirect(d.size)
        bb.put(d)
        bb.rewind()
        GLES20.glPixelStorei(GLES20.GL_UNPACK_ALIGNMENT, 1)
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE, DepthNet.OW, DepthNet.OH, 0, GLES20.GL_LUMINANCE, GLES20.GL_UNSIGNED_BYTE, bb)
    }

    // ---------- OpenGL ----------
    private var progQ = 0; private var progD = 0; private var progW = 0
    private var wPos = 0; private var wMvp = 0; private var wUv = 0; private var wBend = 0; private var wQuad = 0
    private var wBox = 0; private var wRad = 0; private var wGlow = 0; private var wAlpha = 0; private var wShadow = 0; private var wTex = 0
    private val STRIPN = 24
    private lateinit var strip: FloatBuffer
    private lateinit var quad: FloatBuffer
    private var qPos = 0; private var qMvp = 0; private var qUv = 0; private var qA = 0; private var qTex = 0
    private var dPos = 0; private var dK1 = 0; private var dK2 = 0; private var dZ = 0; private var dTex = 0
    private var sw = 1; private var sh = 1; private var fw = 2; private var fh = 2
    private var fbo = 0; private var fboTex = 0
    private var camTex = 0; private var lastCam: Bitmap? = null
    private var lastVer = -1
    private var lastT = System.nanoTime()
    private var curBmp: Bitmap? = null; private var curTex = 0; private var curKey = -1
    private val FULL = floatArrayOf(2f, 0f, 0f, 0f, 0f, 2f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f)

    init {
        setEGLContextClientVersion(2)
        setRenderer(this)
        renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        preserveEGLContextOnPause = true
        loadIcons()
        for (w in wins) spawn(w)
        kb.visible = false; cfg.visible = false
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        progQ = GlUtil.program(VS_Q, FS_Q); progD = GlUtil.program(VS_D, FS_D)
        progW = GlUtil.program(VS_W, FS_W)
        wPos = GLES20.glGetAttribLocation(progW, "aPos"); wMvp = GLES20.glGetUniformLocation(progW, "uMVP")
        wUv = GLES20.glGetUniformLocation(progW, "uUV"); wBend = GLES20.glGetUniformLocation(progW, "uBend")
        wQuad = GLES20.glGetUniformLocation(progW, "uQuad"); wBox = GLES20.glGetUniformLocation(progW, "uBox")
        wRad = GLES20.glGetUniformLocation(progW, "uRad"); wGlow = GLES20.glGetUniformLocation(progW, "uGlow")
        wAlpha = GLES20.glGetUniformLocation(progW, "uA"); wShadow = GLES20.glGetUniformLocation(progW, "uShadow")
        wTex = GLES20.glGetUniformLocation(progW, "uTex")
        wDepth = GLES20.glGetUniformLocation(progW, "uDepth"); wVp = GLES20.glGetUniformLocation(progW, "uVp")
        wCrop = GLES20.glGetUniformLocation(progW, "uCrop"); wOcc = GLES20.glGetUniformLocation(progW, "uOcc")
        qPos = GLES20.glGetAttribLocation(progQ, "aPos"); qMvp = GLES20.glGetUniformLocation(progQ, "uMVP")
        qUv = GLES20.glGetUniformLocation(progQ, "uUV"); qA = GLES20.glGetUniformLocation(progQ, "uA"); qTex = GLES20.glGetUniformLocation(progQ, "uTex")
        dPos = GLES20.glGetAttribLocation(progD, "aPos"); dK1 = GLES20.glGetUniformLocation(progD, "uK1")
        dK2 = GLES20.glGetUniformLocation(progD, "uK2"); dZ = GLES20.glGetUniformLocation(progD, "uZ"); dTex = GLES20.glGetUniformLocation(progD, "uTex")
        quad = ByteBuffer.allocateDirect(32).order(ByteOrder.nativeOrder()).asFloatBuffer()
        quad.put(floatArrayOf(-.5f, -.5f, .5f, -.5f, -.5f, .5f, .5f, .5f)); quad.position(0)
        val sv = FloatArray((STRIPN + 1) * 4)                                   // faixa com varias colunas para curvar a janela
        for (i in 0..STRIPN) { val x = -.5f + i.toFloat() / STRIPN; sv[i * 4] = x; sv[i * 4 + 1] = -.5f; sv[i * 4 + 2] = x; sv[i * 4 + 3] = .5f }
        strip = ByteBuffer.allocateDirect(sv.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        strip.put(sv); strip.position(0)
        for (w in wins) { w.tex = 0; w.tTex = 0; w.dirty = true; w.tDirty = true }
        camTex = 0; lastCam = null; ovTex = 0; curTex = 0; curKey = -1; lastVer = -1; fbo = 0; depthTex = 0; lastDepthVer = -1
        texDims.clear(); lastOvKey = ""
        GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        ui("dock", "addr", "apps", "kb", "cfg", "web")
    }

    override fun onSurfaceChanged(gl: GL10?, w: Int, h: Int) {
        sw = w; sh = h
        buildFbo()
    }

    private fun buildFbo() {
        if (fbo != 0) {
            GLES20.glDeleteFramebuffers(1, intArrayOf(fbo), 0)
            GLES20.glDeleteTextures(1, intArrayOf(fboTex), 0)
        }
        val sc = levels[level]
        fw = ((sw * sc).toInt() / 2) * 2; fh = (sh * sc).toInt()
        fboTex = GlUtil.tex()
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, fw, fh, 0, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null)
        val f = IntArray(1); GLES20.glGenFramebuffers(1, f, 0); fbo = f[0]
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo)
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, fboTex, 0)
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        val ovW = min(fw / 2, 640).coerceAtLeast(64)
        ovBmp = Bitmap.createBitmap(ovW, max(fh * ovW / (fw / 2), 64), Bitmap.Config.ARGB_8888)
        lastOvKey = ""
        rebuild = false
    }

    /** Envia o bitmap para a textura; depois da primeira vez usa atualizacao parcial (bem mais rapida). */
    private fun up(tex: Int, b: Bitmap): Int {
        val t = if (tex == 0) GlUtil.tex() else tex.also { GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, it) }
        val dims = (b.width.toLong() shl 32) or b.height.toLong()
        if (texDims[t] == dims) {
            GLUtils.texSubImage2D(GLES20.GL_TEXTURE_2D, 0, 0, 0, b)
        } else {
            GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, b, 0)
            texDims[t] = dims
        }
        return t
    }

    private fun drawQuad(mvp: FloatArray, tex: Int, u0: Float, v0: Float, us: Float, vs: Float, a: Float) {
        GLES20.glUseProgram(progQ)
        GLES20.glEnableVertexAttribArray(qPos)
        GLES20.glVertexAttribPointer(qPos, 2, GLES20.GL_FLOAT, false, 0, quad)
        GLES20.glUniformMatrix4fv(qMvp, 1, false, mvp, 0)
        GLES20.glUniform4f(qUv, u0, v0, us, vs)
        GLES20.glUniform1f(qA, a)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tex)
        GLES20.glUniform1i(qTex, 0)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
    }

    private fun fillModel(m: FloatArray, w: Win, sx: Float, sy: Float, cy: Float) {
        for (i in 0..2) { m[i] = w.r[i] * sx; m[4 + i] = w.u[i] * sy; m[8 + i] = w.n[i]; m[12 + i] = w.pos[i] + w.u[i] * cy }
        m[3] = 0f; m[7] = 0f; m[11] = 0f; m[15] = 1f
    }

    private fun drawWinQuad(vp: FloatArray, w: Win, tex: Int, bx: Float, by: Float, cy: Float, rad: Float,
                            uvScale: Float, alpha: Float, glow: Float, shadow: Boolean) {
        val pad = if (shadow) .09f else 0f
        val qx = bx + 2f * pad; val qy = by + 2f * pad
        val m = FloatArray(16); val mvp = FloatArray(16)
        fillModel(m, w, qx, qy, cy)
        GLM.multiplyMM(mvp, 0, vp, 0, m, 0)
        GLES20.glUseProgram(progW)
        GLES20.glEnableVertexAttribArray(wPos)
        GLES20.glVertexAttribPointer(wPos, 2, GLES20.GL_FLOAT, false, 0, strip)
        GLES20.glUniformMatrix4fv(wMvp, 1, false, mvp, 0)
        GLES20.glUniform4f(wUv, 0f, 0f, uvScale, 1f)
        GLES20.glUniform1f(wBend, qx * qx / (2f * CURVE_R))
        GLES20.glUniform2f(wQuad, qx, qy)
        GLES20.glUniform2f(wBox, bx, by)
        GLES20.glUniform1f(wRad, rad)
        GLES20.glUniform1f(wGlow, glow)
        GLES20.glUniform1f(wAlpha, alpha)
        GLES20.glUniform1f(wShadow, if (shadow) 1f else 0f)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tex)
        GLES20.glUniform1i(wTex, 0)
        val dn = depth
        val occActive = dn != null && dn.enabled && depthTex != 0 && cam != null
        GLES20.glUniform4f(wVp, curEye * (fw / 2f), 0f, fw / 2f, fh.toFloat())
        GLES20.glUniform4f(wCrop, cu0, cv0, cus, cvs)
        GLES20.glUniform1f(wOcc, if (occActive) 1f else 0f)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, if (depthTex != 0) depthTex else tex)
        GLES20.glUniform1i(wDepth, 1)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, (STRIPN + 1) * 2)
    }

    private fun drawWin(w: Win, vp: FloatArray) {
        val k = .9f + .1f * w.anim.coerceIn(0f, 1f)             // abre crescendo
        val a = w.anim.coerceIn(0f, 1f)
        val ws = w.w * w.scale * k; val hs = w.h * w.scale * k
        if (w.tex != 0) {
            drawWinQuad(vp, w, w.tex, ws, hs, 0f, w.rad * k, w.uScale, a, w.glow, true)      // sombra
            drawWinQuad(vp, w, w.tex, ws, hs, 0f, w.rad * k, w.uScale, a, w.glow, false)
        }
        if (w.hasTitle && w.tTex != 0) {
            val th = w.thM * k
            drawWinQuad(vp, w, w.tTex, ws, th, -(hs / 2f + th / 2f), th / 2f, 1f, a, w.glow * .6f, false)
        }
    }

    private fun drawCursor(vp: FloatArray) {
        val h = curHit
        val pos: FloatArray; val r: FloatArray; val u: FloatArray; val n: FloatArray; val dist: Float
        if (h != null) {
            val w = h.win
            pos = floatArrayOf(rayO[0] + rayD[0] * h.t + w.n[0] * .01f, rayO[1] + rayD[1] * h.t + w.n[1] * .01f, rayO[2] + rayD[2] * h.t + w.n[2] * .01f)
            r = w.r; u = w.u; n = w.n; dist = h.t
        } else {
            pos = floatArrayOf(rayO[0] + rayD[0] * 2.5f, rayO[1] + rayD[1] * 2.5f, rayO[2] + rayD[2] * 2.5f)
            r = R; u = U; n = floatArrayOf(-F[0], -F[1], -F[2]); dist = 2.5f
        }
        val key = (if (pinching) 1000 else 0) + (max(dwell, 0f) * 20f).toInt()
        if (key != curKey || curTex == 0) {
            curKey = key
            val b = curBmp ?: Bitmap.createBitmap(96, 96, Bitmap.Config.ARGB_8888).also { curBmp = it }
            b.eraseColor(0); val c = Canvas(b)
            p.style = Paint.Style.STROKE; p.strokeWidth = 5f; p.color = if (pinching) 0xFF22C55E.toInt() else Color.WHITE
            c.drawCircle(48f, 48f, 26f, p)
            if (dwell > 0f) { p.color = 0xFF9D86FF.toInt(); c.drawArc(RectF(10f, 10f, 86f, 86f), -90f, 360f * dwell / 1.2f, false, p) }
            p.style = Paint.Style.FILL; p.color = if (pinching) 0xFF22C55E.toInt() else Color.WHITE; c.drawCircle(48f, 48f, 6f, p)
            curTex = up(curTex, b)
        }
        val s = (.036f * dist / 2f).coerceIn(.025f, .08f)
        val m = FloatArray(16); val mvp = FloatArray(16)
        for (i in 0..2) { m[i] = r[i] * s; m[4 + i] = u[i] * s; m[8 + i] = n[i]; m[12 + i] = pos[i] }
        m[3] = 0f; m[7] = 0f; m[11] = 0f; m[15] = 1f
        GLM.multiplyMM(mvp, 0, vp, 0, m, 0)
        drawQuad(mvp, curTex, 0f, 0f, 1f, 1f, 1f)
    }

    override fun onDrawFrame(gl: GL10?) {
        val now = System.nanoTime(); val dt = ((now - lastT) / 1e9f).coerceAtMost(.1f); lastT = now
        curFr = frame                                           // fixa camera+maos deste quadro (tudo consistente)
        val es = edState; if (es >= 0) { edState = -1; applyEditable(es == 1) }
        if (pressedT > 0f) { pressedT -= dt; if (pressedT <= 0f) { pressedKey = ""; ui("kb") } }
        if (rebuild) buildFbo()
        // qualidade adaptativa: sobe ou desce a resolucao conforme o tempo de quadro
        acc += dt; frames++; cool -= dt
        if (frames >= 45) {
            val avg = acc / frames; frames = 0; acc = 0f
            if (cool <= 0f) {
                if (avg > .028f && level > 0) { level--; rebuild = true; cool = 3f }
                else if (avg < .0175f && level < levels.size - 1) { level++; rebuild = true; cool = 8f }
            }
        }
        if (now - lastClock > 30_000_000_000L) { lastClock = now; ui("dock") }
        updateHead()

        // a cabeca fica parada no centro: so o giroscopio gira a vista (sem andar pela camera/passos)

        if (now - lastPoll > 30_000_000L) {                  // navegador: no maximo ~30 quadros/s
            lastPoll = now
            val bf = browser.poll()
            if (bf != null && browser.version != lastVer) {
                lastVer = browser.version; browserWin.bmp = bf
                browserWin.uScale = browser.w.toFloat() / bf.width; browserWin.dirty = true
            }
        }
        val cb = cam
        val ew = fw / 2f; val eh = fh.toFloat()
        if (cb != null) {
            cropS = max(ew / cb.width, eh / cb.height)
            val srcW = ew / cropS; val srcH = eh / cropS
            cropL = (cb.width - srcW) / 2f; cropT = (cb.height - srcH) / 2f
            cu0 = cropL / cb.width; cv0 = cropT / cb.height; cus = srcW / cb.width; cvs = srcH / cb.height
            if (cb !== lastCam) { camTex = up(camTex, cb); lastCam = cb }
        }
        if (cb != null && cb !== lastAiCam) {                  // quadro novo: rastreio visual e profundidade
            lastAiCam = cb
            val tms = now / 1_000_000L
            depth?.feed(cb, tms)
        }
        val dn2 = depth
        if (dn2 != null) {
            if (dn2.status != lastDepthStatus) { lastDepthStatus = dn2.status; ui("cfg") }
            val dd = dn2.depth
            if (dn2.enabled && dd != null && dn2.version != lastDepthVer) { lastDepthVer = dn2.version; upDepth(dd) }
        }

        interact(dt)
        if (dirtyIds.isNotEmpty()) {                          // redesenha so as janelas que mudaram
            for (id in dirtyIds) {
                when (id) {
                    "dock" -> drawDock()
                    "addr" -> drawAddr()
                    "apps" -> drawApps()
                    "kb" -> drawKb()
                    "cfg" -> drawCfg()
                }
            }
            for (w in wins) if (dirtyIds.contains(w.id)) w.tDirty = true
            dirtyIds.clear()
        }
        val ovKey = "" + (cam != null) + hands.size + touchMode + pinching + level
        if (ptrs.isNotEmpty() || prevPtrs > 0 || ovKey != lastOvKey) {        // overlay so quando ha mao (ou acabou de sumir)
            lastOvKey = ovKey
            drawOverlay()
            ovBmp?.let { ovTex = up(ovTex, it) }
        }
        prevPtrs = ptrs.size
        for (w in wins) {
            if (!w.visible) continue
            if (w.hasTitle && w.tDirty) { drawTitle(w); w.tBmp?.let { w.tTex = up(w.tTex, it) } }
            if (w.dirty) { w.bmp?.let { w.tex = up(w.tex, it) }; w.dirty = false }
        }
        val hovWin = hoverKey.substringBefore(':')                // animacao de abrir/fechar e brilho ao apontar
        for (w in wins) {
            w.anim += ((if (w.visible) 1f else 0f) - w.anim) * minOf(1f, dt * 10f)
            val gt = if (w.id == hovWin || (gw === w && mode != 0)) 1f else 0f
            w.glow += (gt - w.glow) * minOf(1f, dt * 12f)
        }
        render()
    }

    private fun render() {
        val ew = fw / 2
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo)
        GLES20.glViewport(0, 0, fw, fh)
        GLES20.glClearColor(.02f, .03f, .07f, 1f)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
        val view = FloatArray(16); val proj = FloatArray(16); val vp = FloatArray(16)
        GLM.perspectiveM(proj, 0, 90f, ew.toFloat() / fh, .05f, 60f)
        for (eye in 0..1) {
            GLES20.glViewport(eye * ew, 0, ew, fh)
            curEye = eye
            if (cam != null && camTex != 0) drawQuad(FULL, camTex, cu0, cv0, cus, cvs, 1f)
            val s = if (eye == 0) -ipd / 2f else ipd / 2f
            val ex = head[0] + R[0] * s; val ey = head[1] + R[1] * s; val ez = head[2] + R[2] * s
            GLM.setLookAtM(view, 0, ex, ey, ez, ex + F[0], ey + F[1], ez + F[2], U[0], U[1], U[2])
            GLM.multiplyMM(vp, 0, proj, 0, view, 0)
            val vis = wins.filter { it.anim > .03f }.sortedByDescending {
                val dx = it.pos[0] - ex; val dy = it.pos[1] - ey; val dz = it.pos[2] - ez; dx * dx + dy * dy + dz * dz
            }
            for (w in vis) drawWin(w, vp)
            drawCursor(vp)
            if (ovTex != 0) drawQuad(FULL, ovTex, 0f, 0f, 1f, 1f, 1f)
        }
        // passada final: correcao de distorcao das lentes
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        GLES20.glViewport(0, 0, sw, sh)
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
        GLES20.glUseProgram(progD)
        GLES20.glEnableVertexAttribArray(dPos)
        GLES20.glVertexAttribPointer(dPos, 2, GLES20.GL_FLOAT, false, 0, quad)
        GLES20.glUniform1f(dK1, k1); GLES20.glUniform1f(dK2, k2); GLES20.glUniform1f(dZ, 1f / (1f + k1 + k2))
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, fboTex)
        GLES20.glUniform1i(dTex, 0)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
    }
}
