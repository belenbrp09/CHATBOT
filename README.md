# Chatbot de Comida Peruana + Transcripción de Audio

Proyecto de la práctica: **Sesión 2 — Creación de Chatbots y aplicación de
modelos de OpenAI** (Unidad 2, HDP-TIC, UTP).

## Contenido

- **Parte 1 — Chatbot básico (GPT + Streamlit):** responde preguntas sobre
  comida peruana y mantiene el contexto de la conversación con
  `st.session_state`.
- **Parte 2 — Transcripción de audio (Whisper):** permite subir un archivo
  de audio y obtener su transcripción mediante el modelo `whisper-1`.

## Instalación

```bash
pip install openai streamlit
```

(o bien `pip install -r requirements.txt`)

## Ejecución

```bash
streamlit run app.py
```

## Configuración de la API Key

Puedes ingresar tu `OpenAI API Key` directamente en la barra lateral de la
app, o definirla como variable de entorno antes de ejecutar:

```bash
# PowerShell
$env:OPENAI_API_KEY="tu-api-key"

# bash
export OPENAI_API_KEY="tu-api-key"
```

## Estructura

```
chatbot_peru/
├── app.py              # Aplicación Streamlit (chatbot + transcripción)
├── requirements.txt     # Dependencias
└── README.md            # Este archivo
```
