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
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient

/**
 * Roda um WebView numa tela virtual (Presentation) e entrega os quadros como Bitmap.
 * Assim o vídeo do YouTube (que usa camada de hardware) aparece corretamente.
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

    private class Pres(c: Context, d: Display, val first: String) : Presentation(c, d) {
        lateinit var web: WebView
        override fun onCreate(s: Bundle?) {
            super.onCreate(s)
            web = WebView(context).apply {
                settings.javaScriptEnabled = true
                settings.domStorageEnabled = true
                settings.mediaPlaybackRequiresUserGesture = false
                settings.userAgentString =
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                isFocusable = true
                isFocusableInTouchMode = true
                webViewClient = WebViewClient()
                webChromeClient = WebChromeClient()
                loadUrl(first)
            }
            setContentView(web)
            web.requestFocus()
        }
    }

    fun start(url: String) { pres = Pres(ctx, vd.display, url).also { it.show() } }
    fun load(url: String) { pres?.web?.loadUrl(url) }
    fun back() { pres?.web?.let { if (it.canGoBack()) it.goBack() } }

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


    // ---------- digitacao dentro da pagina (chamar na thread de UI) ----------
    private fun jsq(s: String): String {
        val sb = StringBuilder("'")
        for (c in s) when (c) {
            '\\' -> sb.append("\\\\")
            '\'' -> sb.append("\\'")
            '\n' -> sb.append("\\n")
            else -> sb.append(c)
        }
        return sb.append("'").toString()
    }

    private val ACTIVE = "var e=document.activeElement;while(e&&e.shadowRoot&&e.shadowRoot.activeElement)e=e.shadowRoot.activeElement;"

    fun typeText(s: String) {
        val js = "(function(c){" + ACTIVE + "if(!e)return;var ok=false;try{ok=document.execCommand('insertText',false,c);}catch(x){}" +
            "if(!ok&&('value' in e)){var a=e.selectionStart,z=e.selectionEnd,v=e.value||'';if(a==null){a=v.length;z=a;}" +
            "e.value=v.slice(0,a)+c+v.slice(z);try{e.selectionStart=e.selectionEnd=a+c.length;}catch(x){}" +
            "e.dispatchEvent(new Event('input',{bubbles:true}));}})(" + jsq(s) + ")"
        pres?.web?.evaluateJavascript(js, null)
    }

    fun backspace() {
        val js = "(function(){" + ACTIVE + "if(!e)return;var ok=false;try{ok=document.execCommand('delete');}catch(x){}" +
            "if(!ok&&('value' in e)&&e.value){var a=e.selectionStart,z=e.selectionEnd;if(a==null){a=e.value.length;z=a;}" +
            "if(a==z&&a>0)a--;e.value=e.value.slice(0,a)+e.value.slice(z);try{e.selectionStart=e.selectionEnd=a;}catch(x){}" +
            "e.dispatchEvent(new Event('input',{bubbles:true}));}})()"
        pres?.web?.evaluateJavascript(js, null)
    }

    fun enter() {
        val js = "(function(){" + ACTIVE + "if(!e)return;var o={key:'Enter',code:'Enter',keyCode:13,which:13,bubbles:true,cancelable:true};" +
            "var go=e.dispatchEvent(new KeyboardEvent('keydown',o));e.dispatchEvent(new KeyboardEvent('keypress',o));" +
            "e.dispatchEvent(new KeyboardEvent('keyup',o));" +
            "if(go){if((e.tagName||'')==='TEXTAREA'){document.execCommand('insertText',false,'\\n');}" +
            "else{var f=e.form||(e.closest&&e.closest('form'));if(f){if(f.requestSubmit)f.requestSubmit();else f.submit();}}}})()"
        pres?.web?.evaluateJavascript(js, null)
    }

    /** Diz se o foco da pagina esta num campo de texto (para abrir o teclado sozinho). */
    fun queryEditable(cb: (Boolean) -> Unit) {
        val w = pres?.web ?: return
        val js = "(function(){" + ACTIVE + "if(!e)return false;var t=(e.tagName||'').toLowerCase();if(t==='textarea')return true;" +
            "if(t==='input'){var y=(e.type||'text').toLowerCase();return ['button','checkbox','radio','submit','range','color','file','image','reset'].indexOf(y)<0;}" +
            "return !!e.isContentEditable;})()"
        w.evaluateJavascript(js) { r -> cb(r == "true") }
    }

    fun release() { pres?.dismiss(); vd.release(); reader.close() }
}
