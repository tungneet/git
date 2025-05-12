import streamlit as st
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer
import tempfile
import numpy as np
import scipy.io.wavfile as wavfile
import os
import asyncio
from st_audiorec import st_audiorec

# ✅ Must be first Streamlit command
st.set_page_config(page_title="🗣️ Hinglish Voice Chatbot", layout="centered")

# 🎤 Set sample rate
SAMPLE_RATE = 44100  # st_audiorec uses 44100 Hz

# 🔑 OpenAI Key Input
openai_key = st.text_input("Enter your OpenAI API key:", type="password")
if not openai_key:
    st.warning("Please enter an OpenAI API key to proceed.")
    st.stop()

client = AsyncOpenAI(api_key=openai_key)

# 📥 Save audio to WAV
def save_to_wav(audio_data: np.ndarray):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wavfile.write(temp_file.name, SAMPLE_RATE, audio_data)
    return temp_file.name

# 🧠 Transcribe audio
async def transcribe(file_path):
    with open(file_path, "rb") as f:
        transcript = await client.audio.transcriptions.create(
            file=f,
            model="whisper-1",
            response_format="text"
        )
    return transcript.strip()

# 💬 Generate GPT response
async def generate_reply(prompt):
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a friendly Hinglish customer support agent."},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content.strip()

# 🔊 Speak out response
async def speak(text):
    async with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="onyx",
        input=text,
        instructions="Speak in Indian accent in Hinglish. Use natural, friendly tone.",
        response_format="pcm"
    ) as response:
        await LocalAudioPlayer().play(response)

# 🚀 UI and logic
st.title("🗣️ Hinglish Voice Chatbot")
st.markdown("🎤 Record your message below:")

# 🎙️ Audio recording
audio_bytes = st_audiorec()

if isinstance(audio_bytes, np.ndarray):
    st.success("Audio recorded! Click 'Process Audio' to continue.")
    if st.button("▶️ Process Audio"):
        async def main():
            wav_path = save_to_wav(audio_bytes)
            st.info("Transcribing...")
            user_text = await transcribe(wav_path)
            st.success(f"You said: {user_text}")

            st.info("Generating reply...")
            reply = await generate_reply(user_text)
            st.info(f"🤖 Bot: {reply}")

            await speak(reply)
            os.remove(wav_path)

        asyncio.run(main())

elif audio_bytes is None:
    st.info("Press the mic button to record your message.")
