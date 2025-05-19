# 🤖🎤 Voice Assistant with Conversational RAG

This project is a fully voice-enabled AI assistant built with Python. It uses **OpenAI Whisper** for speech recognition, **LangChain RAG (Retrieval-Augmented Generation)** with chat history awareness for contextual responses, and **Text-to-Speech (TTS)** for replying to the user via audio.

## 🔍 Features

- 🎤 Real-time voice input recording
- 🧠 Conversational AI with memory (history-aware retriever)
- 🔎 Contextual question answering using Retrieval-Augmented Generation
- 🗣️ Natural voice output with TTS
- 🧠 Local sentence embedding using `sentence-transformers`
- 🧾 Custom retriever with redundancy filtering
- 💾 Persistent vector store using ChromaDB

## 🛠️ Technologies Used

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech-to-text
- [LangChain](https://www.langchain.com/) - Conversational RAG framework
- [sentence-transformers](https://www.sbert.net/) - For dense text embeddings
- [Chroma](https://www.trychroma.com/) - Vector database
- [Ollama](https://ollama.com/) - Local LLM model support (e.g., `gemma2`)
- `sounddevice`, `numpy`, `rich`, and more...

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/voice_assistant_rag.git
cd voice_assistant_rag
```

### 2. Set up a virtual environment using Poetry

Ensure Poetry is installed: https://python-poetry.org/docs/#installation

Then install dependencies:

```bash
poetry install
```

Activate the virtual environment:

```bash
poetry shell
```

### 3. Optional: Generate `requirements.txt` for pip-based compatibility

```bash
poetry export -f requirements.txt --without-hashes -o requirements.txt
```

> This is useful for Docker, CI/CD pipelines, or users not using Poetry.

## ▶️ Usage

Run the assistant with:

```bash
python run_voice_assistant_rag_v3.0.py
```

### Workflow

1. Press `Enter` to start recording your voice.
2. Press `Enter` again to stop recording.
3. The assistant will:
   - Transcribe your voice to text
   - Generate a response using a contextual LLM
   - Speak the response back using TTS

## 🧩 Project Structure

- `run_voice_assistant_rag_v3.0.py` - Main entry point for the voice assistant
- `tts.py` - Module to synthesize speech from text
- `redundant_filter_retriever.py` - Custom retriever to reduce redundant documents
- `emb/` - Persistent directory for ChromaDB embeddings

## 🧠 Customization

You can modify the assistant’s personality and verbosity by editing the system prompt in `run_voice_assistant_rag_v3.0.py`:

```python
system_prompt = (
    "You are an assistant for question-answering tasks..."
)
```

To switch LLMs or Whisper models, adjust:

```python
stt = whisper.load_model("base.en")
llm = ChatOllama(model="gemma2")
```

## ❗ Requirements

- Python 3.8+
- A working microphone
- A modern CPU (or GPU for Whisper acceleration)
- Ollama running locally (if using a local LLM)

## 🛡️ License

This project is open source and available under the MIT License.

## 🤝 Contributing

Pull requests, issues, and discussions are welcome! Please follow the standard GitHub flow.
