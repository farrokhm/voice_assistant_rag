import time
import threading
import numpy as np
import whisper
import sounddevice as sd
from queue import Queue
from rich.console import Console
from langchain.chains import ConversationChain
from tts import TextToSpeechService
from redundant_filter_retriever import RedundantFilterRetriever
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_ollama import ChatOllama
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="transformers.models.encodec.modeling_encodec",
)


# list of functions


# Calculate embeddings for documents
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return [self.model.encode(text).tolist() for text in texts]

    def embed_query(self, text):
        return self.model.encode(text).tolist()


def record_audio(stop_event, data_queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.
    Args:
        stop_event (threading.Event): An event that, when set, signals the function to stop recording.
        data_queue (queue.Queue): A queue to which the recorded audio data will be added.
    Returns:
        None
    """

    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)


def transcribe(audio_np: np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.
    Args:
        audio_np (numpy.ndarray): The audio data to be transcribed.
    Returns:
        str: The transcribed text.
    """
    result = stt.transcribe(audio_np, fp16=False)  # Set fp16=True if using a GPU
    text = result["text"].strip()
    return text


def get_llm_response(text: str) -> str:
    """
    Generates a response to the given text using the language model.
    Args:
        text (str): The input text to be processed.
    Returns:
        str: The generated response.
    """
    # response = conversational_rag_chain.invoke({"input": text})
    response = conversational_rag_chain.invoke(
        {"input": text},
        config={
            "configurable": {"session_id": "abc123"}
        },  # constructs a key "abc123" in `store`.
    )  # ["answer"]

    # For debugging purposes: Inspect the structure of the response
    # console.print(f"Response structure: {response}")
    # console.print(f"variable type: {type(response)}")

    # Extract the response text from the 'answer' key
    response_text = response.get("answer", "")
    # response_text = response

    return response_text


def play_audio(sample_rate, audio_array):
    """
    Plays the given audio data using the sounddevice library.

    Args:
        sample_rate (int): The sample rate of the audio data.
        audio_array (numpy.ndarray): The audio data to be played.

    Returns:
        None
    """
    try:
        buffer_size = 2048  # Adjust buffer size if needed
        sd.play(audio_array, sample_rate, blocksize=buffer_size)
        sd.wait()
    except sd.PortAudioError as e:
        console.print(f"[red]Playback error: {e}")


console = Console()
stt = whisper.load_model("base.en")
tts = TextToSpeechService()

# template = """
# You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less
# than 20 words.
# The conversation transcript is as follows:
# {history}
# And here is the user's follow-up: {input}
# Your response:
# """
# PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
# chain = ConversationChain(
#     prompt=PROMPT,
#     verbose=False,
#     memory=ConversationBufferMemory(ai_prefix="Assistant:"),
#     llm=Ollama(),
# )


# Load a pre-trained sentence transformer model locally
# model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer("all-mpnet-base-v2")
embeddings = SentenceTransformerEmbeddings(model)

# vector store
db = Chroma(persist_directory="emb", embedding_function=embeddings)

# retriever = db.as_retriever()
retriever = RedundantFilterRetriever(embeddings=embeddings, chroma=db)
# llm = Ollama(model="gemma2")
llm = ChatOllama(model="gemma2")


# template = """You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less than 50 words.
# Answer the following question based on the provided context
# <context>
# {context}
# </context>

# Question:{input}
# """
# This creates a chain that will combine documents and use the provided template and language model to generate responses.
# prompt = ChatPromptTemplate.from_template(template) # no-memory version of template (single message)


system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


if __name__ == "__main__":
    console.print("[magenta]Assistant started! Press Ctrl+C to exit.")

    try:
        while True:
            console.input(
                "[magenta]Press Enter to start recording, then press Enter again to stop."
            )

            data_queue = Queue()  # type: ignore[var-annotated]
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue),
            )
            recording_thread.start()

            input()
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="earth"):
                    text = transcribe(audio_np)
                console.print(f"[yellow]You: {text}")

                with console.status(
                    "Generating response...", spinner="earth"
                ):  # alternative spinner: "aesthetic"
                    response = get_llm_response(text)
                    sample_rate, audio_array = tts.long_form_synthesize(response)

                # console.print(f"[cyan]Assistant: {response}")
                console.print(f"[cyan]{response}")
                play_audio(sample_rate, audio_array)
            else:
                console.print(
                    "[red]No audio recorded. Please ensure your microphone is working."
                )

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")
