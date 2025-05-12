import asyncio
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import tempfile
import os
import streamlit as st
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer

# Initialize OpenAI client
client = AsyncOpenAI(api_key="add_key_here")

# Audio settings
SAMPLE_RATE = 16000
THRESHOLD = 1000  # adjust as needed
SILENCE_DURATION = 1.5  # seconds of silence to stop recording

# Detects silence to stop recording
def record_until_silence():
    st.write("🎤 Speak now...")
    recording = []
    silence_count = 0
    block_size = 1024

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16') as stream:
        while True:
            block, _ = stream.read(block_size)
            volume = np.linalg.norm(block)
            recording.append(block)

            if volume < THRESHOLD:
                silence_count += block_size / SAMPLE_RATE
                if silence_count >= SILENCE_DURATION:
                    break
            else:
                silence_count = 0

    audio = np.concatenate(recording, axis=0)
    st.write("🔇 Recording stopped.")
    return audio

# Save numpy audio to temp WAV file
def save_to_wav(audio_data):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wavfile.write(temp_file.name, SAMPLE_RATE, audio_data)
    return temp_file.name

# Transcribe using Whisper
async def transcribe(file_path):
    with open(file_path, "rb") as f:
        transcript = await client.audio.transcriptions.create(
            file=f,
            model="whisper-1",
            response_format="text"
        )
    return transcript.strip()

# Generate Hinglish reply using GPT-4o-mini
async def generate_reply(prompt):
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a friendly Hinglish customer support agent."},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content.strip()

# Speak with OpenAI TTS
async def speak(text):
    async with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="onyx",  # Or "echo" for Indian-style voice
        input=text,
        instructions="Speak in Indian accent in Hinglish. Use natural, friendly tone.",
        response_format="pcm"
    ) as response:
        await LocalAudioPlayer().play(response)

# Streamlit interface
st.title("🎤 Hinglish Voice Chatbot")
st.write("Record your voice and interact with the chatbot.")

if st.button("Start Recording"):
    audio = record_until_silence()
    wav_path = save_to_wav(audio)

    try:
        # Transcribe the audio
        user_input = asyncio.run(transcribe(wav_path))
        st.write(f"🗣️ You said: {user_input}")

        # Generate reply from GPT
        reply = asyncio.run(generate_reply(user_input))
        st.write(f"🤖 Bot: {reply}")

        # Speak the reply
        asyncio.run(speak(reply))
    except Exception as e:
        st.write(f"❌ Error: {e}")
    finally:
        os.remove(wav_path)
