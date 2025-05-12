import streamlit as st
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer
import tempfile
import os
import numpy as np
import scipy.io.wavfile as wavfile
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import asyncio

# ✅ MUST be the first Streamlit command
st.set_page_config(page_title="🗣️ Hinglish Voice Chatbot", layout="centered")

# Input OpenAI API Key
openai_key = st.text_input("Enter your OpenAI API key:", type="password")

if not openai_key:
    st.warning("Please enter an OpenAI API key to proceed.")
    st.stop()

client = AsyncOpenAI(api_key=openai_key)
SAMPLE_RATE = 16000

# 🔊 Audio Processor
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []

    def recv(self, frame):
        audio = np.frombuffer(frame.to_ndarray().tobytes(), dtype=np.int16)
        self.audio_frames.append(audio)
        return frame

    def get_audio(self):
        if self.audio_frames:
            return np.concatenate(self.audio_frames)
        return None

# 📥 Save to WAV
def save_to_wav(audio_data):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wavfile.write(temp_file.name, SAMPLE_RATE, audio_data)
    return temp_file.name

# 🎙️ Transcribe Audio
async def transcribe(file_path):
    with open(file_path, "rb") as f:
        transcript = await client.audio.transcriptions.create(
            file=f,
            model="whisper-1",
            response_format="text"
        )
    return transcript.strip()

# 💬 Generate Hinglish Response
async def generate_reply(prompt):
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a friendly Hinglish customer support agent."},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content.strip()

# 🔊 Speak
async def speak(text):
    async with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="onyx",
        input=text,
        instructions="Speak in Indian accent in Hinglish. Use natural, friendly tone.",
        response_format="pcm"
    ) as response:
        await LocalAudioPlayer().play(response)

# 🧠 Main Logic
async def process_audio(audio_data):
    try:
        if audio_data is None:
            st.error("No audio captured.")
            return

        wav_path = save_to_wav(audio_data)
        user_input = await transcribe(wav_path)
        st.success(f"You said: {user_input}")

        reply = await generate_reply(user_input)
        st.info(f"🤖 Bot: {reply}")

        await speak(reply)

        os.remove(wav_path)
    except Exception as e:
        st.error(f"Something went wrong: {e}")

# 🚀 Streamlit UI
st.title("🗣️ Hinglish Voice Chatbot")
st.write("Click the button below to start talking. It will listen, transcribe, reply in Hinglish, and speak it back.")

audio_ctx = st.empty()
processor = AudioProcessor()

result = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=lambda: processor,
    media_stream_constraints={"video": False, "audio": True},
    async_processing=True,
)

if st.button("✅ Stop and Process"):
    if result.state.playing:
        result.stop()
        st.info("Processing your audio...")
        audio_data = processor.get_audio()
        asyncio.create_task(process_audio(audio_data))
