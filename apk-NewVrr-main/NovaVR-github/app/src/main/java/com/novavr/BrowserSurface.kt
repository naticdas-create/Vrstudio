package com.novavr

import android.app.Presentation
import android.content.Context
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.media.ImageReader
import android.os.Bundle
import android.os.SystemClock
import android.view.Display
import android.view.InputDevice
import android.view.MotionEvent
import android.webkit.JavascriptInterface
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import org.json.JSONObject

/**
 * Roda um WebView numa tela virtual (Presentation) e entrega os quadros como Bitmap.
 * Assim o vídeo do YouTube (que usa camada de hardware) aparece corretamente.
 * Tambem digita na pagina (campo de texto em foco) e avisa quando um campo de texto ganha/perde o foco.
 */
class BrowserSurface(private val ctx: Context, val w: Int = 1280, val h: Int = 720) {
    private val reader = ImageReader.newInstance(w, h, PixelFormat.RGBA_8888, 3)
    private val vd = (ctx.getSystemService(Context.DISPLAY_SERVICE) as DisplayManager).createVirtualDisplay(
        "nova-browser", w, h, 160, reader.surface,
        DisplayManager.VIRTUAL_DISPLAY_FLAG_OWN_CONTENT_ONLY or DisplayManager.VIRTUAL_DISPLAY_FLAG_PRESENTATION
    )
    private var pres: Pres? = null
    private var bmp: Bitmap? = null
    private var downTime = 0L
    @Volatile var version = 0
    /** Chamado (em outra thread) quando um campo de texto da pagina ganha (true) ou perde (false) o foco. */
    @Volatile var onTextFocus: ((Boolean) -> Unit)? = null

    private class Bridge(val owner: BrowserSurface) {
        @JavascriptInterface fun focus(f: Boolean) { owner.onTextFocus?.invoke(f) }
    }

    private class Pres(c: Context, d: Display, val first: String, val owner: BrowserSurface) : Presentation(c, d) {
        lateinit var web: WebView
        override fun onCreate(s: Bundle?) {
            super.onCreate(s)
            web = WebView(context).apply {
                settings.javaScriptEnabled = true
                settings.domStorageEnabled = true
                settings.mediaPlaybackRequiresUserGesture = false
                settings.userAgentString =
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                addJavascriptInterface(Bridge(owner), "NovaBridge")
                webViewClient = object : WebViewClient() {
                    override fun onPageFinished(view: WebView?, url: String?) {
                        view?.evaluateJavascript(FOCUS_JS, null)
                    }
                }
                webChromeClient = WebChromeClient()
                loadUrl(first)
            }
            setContentView(web)
        }
    }

    fun start(url: String) { pres = Pres(ctx, vd.display, url, this).also { it.show() } }
    fun load(url: String) { pres?.web?.loadUrl(url) }
    fun back() { pres?.web?.let { if (it.canGoBack()) it.goBack() } }

    private fun js(code: String) { pres?.web?.evaluateJavascript(code, null) }

    /** Digita texto no campo em foco da pagina (chame na thread de UI). */
    fun typeText(s: String) { js("(function(t){" + ACTIVE_JS + SET_JS + INSERT_JS + "})(" + JSONObject.quote(s) + ")") }
    fun typeBackspace() { js("(function(){" + ACTIVE_JS + SET_JS + DELETE_JS + "})()") }
    fun typeEnter() { js("(function(){" + ACTIVE_JS + ENTER_JS + "})()") }

    /** Chame na thread de UI; devolve o quadro mais recente (largura do bitmap pode ser maior que w). */
    fun poll(): Bitmap? {
        val img = reader.acquireLatestImage() ?: return bmp
        val pl = img.planes[0]
        val stride = pl.rowStride / pl.pixelStride
        val b = bmp ?: Bitmap.createBitmap(stride, h, Bitmap.Config.ARGB_8888).also { bmp = it }
        pl.buffer.rewind()
        b.copyPixelsFromBuffer(pl.buffer)
        img.close()
        version++
        return b
    }

    fun touch(action: Int, x: Float, y: Float) {
        val now = SystemClock.uptimeMillis()
        if (action == MotionEvent.ACTION_DOWN) downTime = now
        val ev = MotionEvent.obtain(downTime, now, action, x, y, 0)
        ev.source = InputDevice.SOURCE_TOUCHSCREEN
        pres?.window?.decorView?.dispatchTouchEvent(ev)
        ev.recycle()
    }

    fun release() { pres?.dismiss(); vd.release(); reader.close() }

    private companion object {
        // campo de texto em foco (inclusive dentro de shadow DOM)
        const val ACTIVE_JS = "var e=document.activeElement;while(e&&e.shadowRoot&&e.shadowRoot.activeElement)e=e.shadowRoot.activeElement;if(!e)return;"
        // escreve o valor do jeito que frameworks (React, Polymer) percebem
        const val SET_JS = "function sv(e,v){if(typeof e.value!=='string')return;var P=e.tagName==='TEXTAREA'?HTMLTextAreaElement.prototype:HTMLInputElement.prototype;" +
            "Object.getOwnPropertyDescriptor(P,'value').set.call(e,v);e.dispatchEvent(new Event('input',{bubbles:true}));}"
        const val INSERT_JS = "if(e.isContentEditable){document.execCommand('insertText',false,t);return;}" +
            "if(typeof e.value!=='string')return;var a=e.selectionStart,b=e.selectionEnd;if(a==null){a=b=e.value.length;}" +
            "sv(e,e.value.slice(0,a)+t+e.value.slice(b));try{e.selectionStart=e.selectionEnd=a+t.length;}catch(x){}"
        const val DELETE_JS = "if(e.isContentEditable){document.execCommand('delete');return;}" +
            "if(typeof e.value!=='string')return;var a=e.selectionStart,b=e.selectionEnd;if(a==null){a=b=e.value.length;}" +
            "if(a===b){if(a===0)return;a=a-1;}sv(e,e.value.slice(0,a)+e.value.slice(b));try{e.selectionStart=e.selectionEnd=a;}catch(x){}"
        const val ENTER_JS = "var o={key:'Enter',code:'Enter',keyCode:13,which:13,bubbles:true,cancelable:true};" +
            "var go=e.dispatchEvent(new KeyboardEvent('keydown',o));e.dispatchEvent(new KeyboardEvent('keypress',o));e.dispatchEvent(new KeyboardEvent('keyup',o));" +
            "if(go&&e.form){if(e.form.requestSubmit)e.form.requestSubmit();else e.form.submit();}"
        // avisa o app quando um campo de texto ganha/perde o foco (abre/fecha o teclado sozinho)
        const val FOCUS_JS = "(function(){if(window.__nvf)return;window.__nvf=1;" +
            "function ed(t){return !!t&&((t.tagName==='INPUT'&&!/^(button|checkbox|radio|submit|image|range|color|file|reset)/i.test(t.type||''))||t.tagName==='TEXTAREA'||t.isContentEditable);}" +
            "document.addEventListener('focusin',function(ev){NovaBridge.focus(ed(ev.target));},true);" +
            "document.addEventListener('focusout',function(){setTimeout(function(){NovaBridge.focus(ed(document.activeElement));},60);},true);})();"
    }
}
