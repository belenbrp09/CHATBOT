"""
Chatbot de Comida Peruana + Chat por Voz (STT + TTS)
----------------------------------------------------
Cumple con las condiciones de la práctica (Sesión 2 - Unidad 2):

Parte 1: Chatbot básico (Chat Completions) que responde sobre comida
         peruana, con interfaz en Streamlit y contexto de conversación
         mantenido en st.session_state.

Parte 2: Chat por voz: el usuario habla por el micrófono, el audio se
         transcribe con un modelo de la familia Whisper (Groq) y el
         chatbot responde por voz usando un modelo de texto-a-voz (TTS).

Pipeline del chat por voz:
    Micrófono (st.audio_input)
        -> Whisper (speech-to-text, Groq)
        -> GPT (chat completion, Groq)
        -> TTS (text-to-speech) y reproducción automática

TTS: por defecto se usa Groq (canopylabs/orpheus-v1-english), que lee
     texto en español con un acento inglés. Si ingresas una OpenAI API
     Key (opcional), se usa `tts-1`, con voces multilingües que suenan
     natural en español.

Proveedor: Groq (API compatible con la librería `openai`; solo cambia
la URL base y los nombres de modelo). Tu API Key de Groq empieza con
"gsk_...".

Requisitos previos:
    pip install -r requirements.txt   (streamlit>=1.38, openai)

Ejecución:
    streamlit run app.py

Nota: el micrófono del navegador solo funciona en localhost o HTTPS.
"""

import base64
import io
import re
import wave

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Chatbot Comida Peruana", page_icon="🇵🇪", layout="centered")

# ---------------------------------------------------------------------------
# Configuración de modelos
# ---------------------------------------------------------------------------
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
CHAT_MODEL = "openai/gpt-oss-20b"          # Chat (Groq)
STT_MODEL = "whisper-large-v3-turbo"       # Voz -> texto (Groq, rápido y multilingüe)

# TTS con Groq (un solo proveedor / una sola key).
# Orpheus es modelo en inglés: lee español con acento, pero funciona.
GROQ_TTS_MODEL = "canopylabs/orpheus-v1-english"
GROQ_TTS_VOICES = ["troy", "autumn", "jessica", "leo", "stella"]
GROQ_TTS_MAX_CHARS = 200   # límite de caracteres por petición de Orpheus

# TTS con OpenAI (opcional): voces multilingües, suenan natural en español.
OPENAI_TTS_MODEL = "tts-1"
OPENAI_TTS_VOICES = ["nova", "alloy", "shimmer", "echo", "onyx", "fable"]
OPENAI_TTS_MAX_CHARS = 4000

with st.sidebar:
    st.header("⚙️ Configuración")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Empieza con 'gsk_...'. También puedes definirla como variable de entorno GROQ_API_KEY",
    )
    openai_key = st.text_input(
        "OpenAI API Key (opcional)",
        type="password",
        help="Solo para la voz del chatbot. Si la ingresas, se usa tts-1 (español natural). "
             "Si no, la voz sale de Groq (Orpheus, con acento inglés).",
    )
    st.markdown("---")

    use_openai_tts = bool(openai_key)
    voice = st.selectbox(
        "🎚️ Voz del chatbot",
        options=OPENAI_TTS_VOICES if use_openai_tts else GROQ_TTS_VOICES,
        index=0,
    )
    st.markdown("---")
    st.markdown(
        "**Modelos usados**\n"
        f"- Chat: `{CHAT_MODEL}` (Groq)\n"
        f"- Voz → texto: `{STT_MODEL}` (Groq)\n"
        f"- Texto → voz: `{OPENAI_TTS_MODEL if use_openai_tts else GROQ_TTS_MODEL}`"
    )
    st.markdown("---")
    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if "mic_input" in st.session_state:
            st.session_state["mic_input"] = None
        st.rerun()

client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL) if api_key else None
openai_client = OpenAI(api_key=openai_key) if openai_key else None

st.title("🇵🇪 Chatbot de Comida Peruana")
st.caption("Sesión 2 · Creación de Chatbots y aplicación de modelos de OpenAI")

tab_chat, tab_audio = st.tabs(["💬 Chatbot", "🎙️ Chat por voz"])

SYSTEM_PROMPT = (
    "Eres un asistente experto en gastronomía peruana. Responde de forma "
    "clara, amable y concisa únicamente sobre platos, ingredientes, historia "
    "y recomendaciones relacionadas con la comida del Perú. Si te preguntan "
    "algo fuera de ese tema, indícalo cordialmente y redirige la conversación "
    "hacia la comida peruana."
)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]


def render_history():
    """Muestra el historial de la conversación (omite el system prompt)."""
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ---------------------------------------------------------------------------
# Funciones auxiliares para el TTS
# ---------------------------------------------------------------------------
def split_for_tts(text: str, max_chars: int) -> list:
    """Divide un texto largo en fragmentos de como máximo `max_chars`."""
    parts = re.split(r"(?<=[.!?;:,\n])\s+", text.strip())
    chunks, buf = [], ""
    for part in parts:
        part = part.strip()
        while len(part) > max_chars:  # frase demasiado larga: cortar por palabras
            if buf:
                chunks.append(buf)
                buf = ""
            cut = part.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            chunks.append(part[:cut].strip())
            part = part[cut:].strip()
        candidate = (buf + " " + part).strip()
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            buf = part
    if buf:
        chunks.append(buf)
    return chunks or [text[:max_chars]]


def concat_wavs(wav_list: list) -> bytes:
    """Une varios audios WAV (mismos parámetros) en un solo WAV."""
    params, frames = None, b""
    for data in wav_list:
        with wave.open(io.BytesIO(data), "rb") as w:
            p = w.getparams()
            f = w.readframes(w.getnframes())
        if params is None:
            params = p
        elif (p.nchannels, p.sampwidth, p.framerate) != (
            params.nchannels, params.sampwidth, params.framerate
        ):
            return wav_list[0]  # parámetros distintos: devolver el primero
        frames += f
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setparams(params)
        w.writeframes(frames)
    return out.getvalue()


def synthesize_speech(text: str, voice_id: str) -> bytes:
    """Genera el audio (bytes WAV) de `text` con la voz configurada."""
    if use_openai_tts:
        resp = openai_client.audio.speech.create(
            model=OPENAI_TTS_MODEL, voice=voice_id, input=text,
            response_format="wav",
        )
        return resp.content
    # Groq (Orpheus): máx. 200 caracteres por petición -> fragmentar y unir
    chunks = split_for_tts(text, GROQ_TTS_MAX_CHARS)
    audios = [
        client.audio.speech.create(
            model=GROQ_TTS_MODEL, voice=voice_id, input=c, response_format="wav"
        ).content
        for c in chunks
    ]
    return audios[0] if len(audios) == 1 else concat_wavs(audios)


def autoplay_audio(wav_bytes: bytes):
    """Reproduce el audio automáticamente en el navegador."""
    b64 = base64.b64encode(wav_bytes).decode()
    st.markdown(
        f'<audio autoplay src="data:audio/wav;base64,{b64}"></audio>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# PARTE 1: Chatbot de texto
# ---------------------------------------------------------------------------
with tab_chat:
    st.subheader("Pregúntame sobre comida peruana")
    render_history()

    user_input = st.chat_input("Escribe tu pregunta sobre comida peruana...")

    if user_input:
        if not client:
            st.error("Ingresa tu Groq API Key en la barra lateral para continuar.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    response = client.chat.completions.create(
                        model=CHAT_MODEL,
                        messages=st.session_state.messages,
                    )
                    reply = response.choices[0].message.content
                    st.markdown(reply)

            st.session_state.messages.append({"role": "assistant", "content": reply})

# ---------------------------------------------------------------------------
# PARTE 2: Chat por voz  (micrófono -> Whisper -> GPT -> TTS)
# ---------------------------------------------------------------------------
with tab_audio:
    st.subheader("Habla con el chatbot")
    st.write(
        "Graba tu mensaje con el micrófono 🎤. El chatbot lo transcribe, "
        "responde y te contesta con voz."
    )
    render_history()

    recorded = st.audio_input(
        "Graba tu pregunta sobre comida peruana", key="mic_input"
    )

    if recorded is not None:
        st.audio(recorded)

        if st.button("📤 Enviar mensaje de voz"):
            if not client:
                st.error("Ingresa tu Groq API Key en la barra lateral para continuar.")
            else:
                audio_bytes = (
                    recorded.getvalue() if hasattr(recorded, "getvalue") else recorded.read()
                )

                # 1) Voz -> texto (Whisper en Groq)
                with st.spinner("🎧 Transcribiendo tu voz..."):
                    transcript = client.audio.transcriptions.create(
                        model=STT_MODEL,
                        file=("mensaje.wav", audio_bytes),
                        language="es",
                    )
                user_text = transcript.text.strip()

                if not user_text:
                    st.warning("No se detectó voz en la grabación. Intenta de nuevo.")
                else:
                    st.session_state.messages.append({"role": "user", "content": user_text})
                    with st.chat_message("user"):
                        st.markdown(user_text)

                    # 2) Texto -> respuesta (Chat Completions)
                    with st.spinner("Pensando..."):
                        response = client.chat.completions.create(
                            model=CHAT_MODEL,
                            messages=st.session_state.messages,
                            max_tokens=400,  # respuestas cortas: ideales para voz
                        )
                        reply = response.choices[0].message.content
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    with st.chat_message("assistant"):
                        st.markdown(reply)

                    # 3) Respuesta -> voz (TTS) y reproducción automática
                    with st.spinner("🔊 Generando voz..."):
                        try:
                            wav_bytes = synthesize_speech(reply, voice)
                        except Exception as e:
                            wav_bytes = None
                            err_msg = str(e)
                    if wav_bytes is not None:
                        autoplay_audio(wav_bytes)
                        st.audio(wav_bytes, format="audio/wav")
                    elif err_msg and "terms" in err_msg.lower():
                        st.error(
                            "La voz de Groq (Orpheus) requiere aceptar sus términos: "
                            "entra a https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english "
                            "con tu cuenta y acepta el aviso. Alternativa: ingresa una OpenAI API Key "
                            "en la barra lateral para usar la voz tts-1 en español."
                        )
                    elif err_msg:
                        st.error(f"Error al generar la voz: {err_msg}")