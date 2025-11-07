"""
Interfaz gráfica de Gradio para transcripción de voz con Whisper.

Este módulo define una aplicación web sencilla que permite al usuario
transcribir audio grabado desde su micrófono utilizando el modelo
``faster-whisper``. Además, se muestra un pequeño chat que replica
la transcripción y puede responder con eco a los mensajes del usuario.

Requisitos:
    pip install gradio faster-whisper ctranslate2 tokenizers

El código está organizado en varias secciones: carga perezosa del modelo,
generación de burbujas de chat en HTML, funciones de acción para enviar
texto o audio y la definición de la interfaz con sus componentes.

"""

import os
import html
from typing import Any, Optional
import gradio as gr

try:
    # ``llama_cpp`` permite cargar modelos en formato GGUF para la familia LLaMA.
    from llama_cpp import Llama  # type: ignore
except Exception:
    Llama = None

# Almacena en caché el modelo LLaMA para evitar recargas repetitivas.
LLAMA_CACHE: dict[str, Optional[object]] = {"obj": None, "path": None}

# Ruta por defecto al modelo LLaMA en formato GGUF.  Se basa en la
# ubicación de este archivo, de modo que si colocas el modelo junto al
# script (por ejemplo ``gpt-oss-20b-MXFP4.gguf``), se podrá cargar sin
# necesidad de escribir la ruta completa en la interfaz. Puedes
# modificar este nombre según corresponda.
_BASE_DIR = os.path.dirname(__file__)
DEFAULT_LLAMA_MODEL_PATH = os.path.join(_BASE_DIR, "gpt-oss-20b-MXFP4.gguf")

def load_llama(model_path: str) -> "Llama":
    """Carga un modelo LLaMA en formato GGUF.

    Usa un mecanismo de caché para reutilizar la instancia si ya se cargó
    previamente con la misma ruta. Requiere la biblioteca ``llama-cpp-python``.

    Args:
        model_path: ruta al archivo ``.gguf`` del modelo.

    Returns:
        Instancia de ``Llama`` configurada para inferencia.

    Raises:
        RuntimeError: si la biblioteca ``llama_cpp`` no está instalada.
        FileNotFoundError: si no existe el archivo de modelo.
    """
    if Llama is None:
        raise RuntimeError(
            "No se encontró llama_cpp. Instala llama-cpp-python para usar modelos LLaMA (formato GGUF)."
        )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No existe el archivo de modelo: {model_path}")
    # Reutiliza la instancia si la ruta coincide
    if LLAMA_CACHE["obj"] is not None and LLAMA_CACHE["path"] == model_path:
        return LLAMA_CACHE["obj"]  # type: ignore[return-value]
    # Crea nueva instancia con contexto de 2048 tokens (puedes ajustar n_ctx)
    # ``n_gpu_layers=-1`` intentará utilizar la GPU (CUDA) para todas
    # las capas si se compiló con soporte CUDA. De lo contrario, se
    # ejecutará en CPU sin error. Puedes ajustar ``n_gpu_layers`` y
    # ``n_ctx`` según tus recursos.
    llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=-1)
    LLAMA_CACHE["obj"] = llm
    LLAMA_CACHE["path"] = model_path
    return llm

# Prompt por defecto para la interacción con LLaMA.
LLAMA_SYSTEM_PROMPT = (
    "Eres un asistente de inteligencia artificial especializado en mantenimiento SAP "
    "y operaciones en Colbún. Proporciona respuestas útiles y precisas en español."
)

def generate_llama_response(
    model_path: str, history: list[dict[str, str]], user_text: str
) -> str:
    """Genera una respuesta usando un modelo LLaMA.

    Construye un prompt basado en el historial de mensajes y utiliza el modelo
    especificado para generar una respuesta. Cada mensaje del historial se
    etiqueta como ``User:`` o ``Assistant:`` para guiar a la red en la
    conversación.

    Args:
        model_path: ruta al archivo GGUF.
        history: historial de mensajes previos.
        user_text: texto más reciente del usuario.

    Returns:
        Respuesta generada por el modelo LLaMA.
    """
    llm = load_llama(model_path)
    # Construye la conversación como instrucciones para el modelo
    conv_lines: list[str] = [LLAMA_SYSTEM_PROMPT.strip(), ""]
    for msg in history:
        role = msg.get("role")
        text = msg.get("text", "").strip()
        if not text:
            continue
        if role == "user":
            conv_lines.append(f"Usuario: {text}")
        else:
            conv_lines.append(f"Asistente: {text}")
    # Añade el último mensaje del usuario
    conv_lines.append(f"Usuario: {user_text.strip()}")
    conv_lines.append("Asistente: ")
    prompt = "\n".join(conv_lines)
    # Invoca el modelo. Configura un número razonable de tokens y parada.
    result = llm(
        prompt,
        max_tokens=256,
        stop=["Usuario:"],
        temperature=0.7,
    )
    # Extrae la respuesta del campo ``choices[0]['text']``. Algunas versiones
    # devuelven ``choices`` como lista de dicts.
    raw_text = ""
    try:
        raw_text = result["choices"][0]["text"]
    except Exception:
        # Fallback para versiones antiguas que devuelven una cadena directa
        raw_text = str(result)
    answer = raw_text.strip()
    return answer

try:
    # ``faster_whisper`` es opcional. Si no está instalado, definimos
    # ``WhisperModel`` como ``None`` y mostraremos un error amigable al usarla.
    from faster_whisper import WhisperModel  # type: ignore
except Exception:
    WhisperModel = None


# -- Modelo global en caché ---------------------------------------------------
MODEL_CACHE: dict[str, object | None] = {"obj": None, "size": None}

# Contexto previo (prompt) utilizado para orientar la transcripción al
# dominio de Colbún en Chile. Añade palabras clave frecuentes para
# mejorar la inferencia cuando se usan modelos multilingües.
PROMPT_CL = (
    "Contexto chileno, términos: Colbún, mantenimiento, SAP, orden, "
    "notificación, equipo, llenar, campos, transacción, horas, trabajo, "
    "necesito, ayuda, ayúdame."
)


def load_model(size: str) -> WhisperModel:
    """Cargar y almacenar en caché un modelo de Whisper.

    Esta función gestiona la carga perezosa de modelos Whisper. Si un modelo
    del mismo tamaño ya está cargado, se reutiliza. Si ``faster_whisper`` no
    está instalado se lanza una excepción informativa.

    Args:
        size: nombre del modelo (por ejemplo, ``"base"`` o ``"small"``).

    Returns:
        Una instancia de ``WhisperModel`` preparada para la transcripción.

    Raises:
        RuntimeError: si ``faster_whisper`` no está instalado.
    """
    if WhisperModel is None:
        raise RuntimeError(
            "Dependencia faltante: faster-whisper no está instalado. "
            "Ejecute 'pip install faster-whisper ctranslate2 tokenizers'."
        )

    # Reutiliza el modelo en caché si el tamaño coincide.
    if MODEL_CACHE["obj"] is not None and MODEL_CACHE["size"] == size:
        return MODEL_CACHE["obj"]  # type: ignore[return-value]

    # Intenta cargar en GPU (FP16) y si falla recurre a la CPU.
    try:
        model = WhisperModel(size, device="cuda", compute_type="float16")
        print(f"[Whisper] Cargado modelo '{size}' en GPU (FP16).")
    except Exception:
        model = WhisperModel(size, device="cpu")
        print(f"[Whisper] Cargado modelo '{size}' en CPU.")

    # Guarda en caché para llamadas posteriores.
    MODEL_CACHE["obj"] = model
    MODEL_CACHE["size"] = size
    return model


# -- Estilos e interfaz de chat ------------------------------------------------
CSS = """
<style>
  :root { --blue:#1e90ff; --light:#ffffff; --text:#111; }
  #chatwrap { background: var(--blue); padding: 0; height: 100%; }
  .chat-scroll { height: 100%; overflow-y: auto; padding: 8px 0 8px 0; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; padding: 6px 14px; }
  .bubble {
    display:inline-block; padding:10px 12px; border-radius:16px; max-width: 720px;
    word-wrap: break-word; font-family: system-ui, sans-serif; font-size: 14px;
    line-height: 1.35; background:#ffffff !important; color:#111 !important;
  }
  .user { justify-self: end; }
  .bot  { justify-self: start; }
  /* Asegura que el texto interno use color oscuro */
  .bubble * { color:#111 !important; }
  .header { background:#ffffff !important; color:#111 !important;
            padding:10px 16px; border-bottom:1px solid #eee; font-weight:700; }
  .small  { font-size: 12px; color:#333 !important; }
</style>
"""


def render_chat(history: list[dict[str, str]]) -> str:
    """Transforma el historial de conversación en HTML con burbujas.

    Args:
        history: lista de mensajes, cada uno con las claves ``role`` y ``text``.

    Returns:
        Cadena HTML que contiene el chat con burbujas de usuario y bot.
    """
    rows: list[str] = []
    for msg in history:
        role_class = "user" if msg.get("role") == "user" else "bot"
        # Escape posibles etiquetas HTML en el texto para evitar XSS.  Usamos
        # html.escape en lugar de depender de utilidades internas de Gradio,
        # ya que ``gr.utils.sanitize_html`` no está disponible en todas las
        # versiones de la biblioteca.
        safe_text = html.escape(msg.get("text", ""))
        rows.append(
            f'<div class="row"><div class="bubble {role_class}">{safe_text}</div></div>'
        )
    return (
        CSS
        + '<div id="chatwrap">'
        + '<div class="header">Chat</div>'
        + f'<div class="chat-scroll" id="scroll">{"".join(rows)}</div>'
        + '</div>'
    )


# -- Acciones ---------------------------------------------------------------
def send_text_enter(
    user_text: str, echo: bool, history: list[dict[str, str]]
) -> tuple[Any, list[dict[str, str]], str, str]:
    """Gestiona el envío de texto desde la caja de entrada.

    El texto del usuario se añade al historial. Si ``echo`` es ``True`` se
    simula una respuesta del bot repitiendo el mismo texto. El valor de la
    caja de texto se limpia en cada llamada.

    Args:
        user_text: texto introducido por la persona usuaria.
        echo: si debe responder con eco de forma automática.
        history: historial acumulado de mensajes.

    Returns:
        - Update para limpiar la caja de texto.
        - Historial actualizado.
        - Cadena HTML para renderizar el chat.
        - Mensaje de estado.
    """
    text = (user_text or "").strip()
    if not text:
        return gr.update(value=""), history, render_chat(history), "Listo."
    history.append({"role": "user", "text": text})
    if echo:
        history.append({"role": "bot", "text": text})
    return gr.update(value=""), history, render_chat(history), "Listo."


def transcribe(
    audio_path: str,
    model_size: str,
    language: str,
    patience: float,
    beam: int,
    min_sil_ms: int,
    pad_ms: int,
    echo: bool,
    history: list[dict[str, str]],
    tts_enabled: bool,
    llama_model_path: Optional[str] = None,
    use_llama: bool = False,
) -> tuple[list[dict[str, str]], str, str]:
    """Transcribe un clip de audio usando un modelo Whisper.

    Este callback se ejecuta al pulsar el botón *Transcribir* en la interfaz.
    Recupera el modelo (con caché), procesa el audio y añade la
    transcripción al historial. En caso de error se captura la excepción
    y se informa en el chat.

    Args:
        audio_path: ruta local al archivo de audio grabado o subido.
        model_size: tamaño del modelo Whisper a usar.
        language: abreviatura ISO del idioma (``"es"`` o ``"en"``).
        patience: hiperparámetro de decodificación de Whisper.
        beam: número de hipótesis en la búsqueda beam.
        min_sil_ms: silencio mínimo en ms para filtrar VAD.
        pad_ms: ms de relleno adicionales que se incluyen al final.
        echo: si se debe responder con eco al usuario.
        history: historial acumulado de mensajes.
        tts_enabled: indicador para TTS (no implementado actualmente).

    Returns:
        - Historial actualizado.
        - Cadena HTML con el chat renderizado.
        - Mensaje de estado.
    """
    if not audio_path:
        history.append({"role": "bot", "text": "No se capturó audio."})
        return history, render_chat(history), "No se capturó audio."
    try:
        model = load_model(model_size)

        # Realiza la transcripción con filtrado VAD.
        segments, info = model.transcribe(
            audio_path,
            language=language or None,
            beam_size=int(beam),
            patience=float(patience),
            temperature=[0.0],
            vad_filter=True,
            vad_parameters=dict(
                min_speech_duration_ms=200,
                max_speech_duration_s=60,
                min_silence_duration_ms=int(min_sil_ms),
                speech_pad_ms=int(pad_ms),
            ),
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=True,
            initial_prompt=PROMPT_CL,
        )
        # Combina todas las porciones en una cadena de texto.
        text = "".join(seg.text for seg in segments).strip()
        if not text:
            text = "No detecté voz, intenta nuevamente."

        # Añade la transcripción al historial como mensaje del usuario.
        history.append({"role": "user", "text": text})
        # Responder usando el modelo LLaMA si está habilitado y se suministró ruta
        if use_llama and llama_model_path:
            try:
                answer = generate_llama_response(llama_model_path, history, text)
                history.append({"role": "bot", "text": answer})
            except Exception as ll_exc:
                history.append({"role": "bot", "text": f"[Error LLaMA] {ll_exc}"})
        elif echo:
            # En modo eco, repetimos el texto como respuesta del bot
            history.append({"role": "bot", "text": text})

        # Por ahora no se implementa TTS en el navegador; se reserva para futuro.
        return history, render_chat(history), "Listo."
    except Exception as exc:
        history.append({"role": "bot", "text": f"[Error] {exc}"})
        return history, render_chat(history), "Error en transcripción."


def clear_audio() -> None:
    """Callback que reinicia el componente de audio.

    Al devolver ``None`` en un ``gr.Audio`` se borra la grabación previa.
    """
    return None


# -- Construcción de la interfaz ------------------------------------------------
with gr.Blocks(
    title="Interfaz Colbún — Micrófono sin modelo (Gradio)"
) as demo:
    # Estado compartido del historial y del mensaje de estado
    history = gr.State([])
    status = gr.State("Listo.")

    with gr.Row():
        # --- Sidebar con configuración ---
        with gr.Column(scale=0, min_width=300, elem_classes=["sidebar"]):
            gr.Markdown("### Interfaz IA (sin modelo)")
            gr.Markdown("Colbún", elem_classes=["small"])
            gr.Markdown("**System Prompt (solo UI)**", elem_classes=["small"])
            gr.Textbox(
                value=(
                    "Modo interfaz: sin llamadas a modelo.\n"
                    "La voz se transcribe localmente (Whisper) y se muestra en el chat."
                ),
                lines=7,
                interactive=False,
                label=""
            )
            # Selección de tamaño del modelo Whisper
            whisper_size = gr.Dropdown(
                ["tiny", "base", "small", "medium", "large-v3"],
                value="base",
                label="Modelo Whisper",
            )
            # Opciones de idioma y parámetros de decodificación
            language_dd = gr.Dropdown(
                ["es", "en"], value="es", label="Idioma"
            )
            patience_slider = gr.Slider(
                minimum=0.2,
                maximum=2.0,
                value=1.2,
                step=0.1,
                label="Paciencia",
            )
            beam_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=8,
                step=1,
                label="Beam",
            )
            silence_ms_slider = gr.Slider(
                minimum=100,
                maximum=1200,
                value=400,
                step=50,
                label="Silencio mínimo (ms)",
            )
            pad_ms_slider = gr.Slider(
                minimum=0,
                maximum=400,
                value=120,
                step=10,
                label="Speech pad (ms)",
            )
            # Configuración del modelo LLaMA (opcional)
            gr.Markdown("**Modelo LLaMA (GGUF)**", elem_classes=["small"])
            llama_model_path = gr.Textbox(
                value=DEFAULT_LLAMA_MODEL_PATH,
                label="Ruta modelo GGUF (.gguf)",
                placeholder="/ruta/al/modelo.gguf",
            )
            use_llama = gr.Checkbox(
                value=False,
                label="Usar modelo LLaMA para responder",
            )
            gr.Markdown("**Umbral RMS** *(referencial)*", elem_classes=["small"])
            rms_slider = gr.Slider(
                minimum=100, maximum=3000, value=500, step=1, label="Umbral RMS"
            )
            gr.Markdown("**Silencio (seg)**", elem_classes=["small"])
            silence = gr.Number(value=1.5, precision=1, label="Silencio (seg)")
            tts = gr.Checkbox(value=False, label="Leer eco (TTS)")
            echo = gr.Checkbox(value=True, label="Responder con eco (sin modelo)")
            gr.Markdown("No se realizan llamadas a backend.", elem_classes=["small"])

        # --- Columna principal: chat y controles de audio ---
        with gr.Column(scale=1):
            chat_html = gr.HTML(render_chat([]), elem_id="chat")
            with gr.Row():
                txt = gr.Textbox(
                    placeholder="Escribe o usa el micrófono…",
                    scale=1,
                    label="Mensaje",
                )
                mic_btn = gr.Button("🎙️", size="sm")  # icono de micrófono
                stop_btn = gr.Button("⏹", size="sm")
            with gr.Row():
                audio = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Audio",
                )
                trans_btn = gr.Button("Transcribir", variant="primary")
            status_lbl = gr.Markdown("Listo.", elem_classes=["small"])

    # -- Envío de texto mediante la tecla Enter --
    txt.submit(
        fn=send_text_enter,
        inputs=[txt, echo, history],
        outputs=[txt, history, chat_html, status_lbl],
    )

    # -- Botón de transcripción de audio --
    trans_btn.click(
        fn=transcribe,
        inputs=[
            audio,
            whisper_size,
            language_dd,
            patience_slider,
            beam_slider,
            silence_ms_slider,
            pad_ms_slider,
            echo,
            history,
            tts,
            llama_model_path,
            use_llama,
        ],
        outputs=[history, chat_html, status_lbl],
    )

    # -- Botón para limpiar audio grabado --
    stop_btn.click(fn=clear_audio, inputs=None, outputs=[audio])

    # Nota: el botón de micrófono actúa únicamente como icono visual. La grabación
    # se controla desde el componente de audio de Gradio.


if __name__ == "__main__":  # pragma: no cover
    # Ejecuta la aplicación usando parámetros configurables a través de
    # variables de entorno. Por defecto se liga a 127.0.0.1 para que pueda
    # abrirse en el navegador local (evitando 0.0.0.0 que a veces se bloquea
    # en ciertos entornos). Si se define la variable SHARE=true el enlace
    # público de Gradio estará disponible.
    port: int = int(os.getenv("PORT", "8080"))
    # Al definir SHARE=true el servidor generará un enlace externo accesible
    # desde cualquier navegador. Por defecto SHARE es false.
    share_flag: bool = os.getenv("SHARE", "false").lower() == "true"
    # Permitir sobreescribir el nombre del servidor via SERVER_NAME, por
    # ejemplo 'localhost'. Cuando ``share_flag`` es True dejamos ``server_name``
    # como None para que Gradio gestione el dominio.
    default_server = "127.0.0.1"
    server_name_env = os.getenv("SERVER_NAME", default_server)
    server_name_param = None if share_flag else server_name_env
    demo.queue().launch(
        server_name=server_name_param,
        server_port=port,
        share=share_flag,
    )
