import streamlit as st
import asyncio
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import tempfile
import os
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer

# Streamlit app setup
st.title("🎤 Hinglish Voice Chatbot")
st.write("Speak to the bot in English or Hinglish and get a friendly response!")

# API Key Input (for development)
if 'OPENAI_API_KEY' not in st.secrets:
    api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    if not api_key:
        st.warning("Please enter your OpenAI API key to continue")
        st.stop()
else:
    api_key = st.secrets["OPENAI_API_KEY"]

# Initialize OpenAI client
client = AsyncOpenAI(api_key=api_key)

# Audio settings
SAMPLE_RATE = 16000
THRESHOLD = 1000
SILENCE_DURATION = 1.5

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Detects silence to stop recording
def record_until_silence():
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

    return np.concatenate(recording, axis=0)

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

# Generate Hinglish reply
async def generate_reply(prompt):
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a friendly Hinglish customer support agent."},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content.strip()

# Speak with OpenAI TTS
async def speak(text):
    async with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="onyx",
        input=text,
        instructions="Speak in Indian accent in Hinglish. Use natural, friendly tone.",
        response_format="pcm"
    ) as response:
        await LocalAudioPlayer().play(response)

# Main chat function
async def run_chat():
    status = st.empty()
    status.info("🎤 Speak now... (Recording)")
    
    audio = record_until_silence()
    status.success("🔇 Recording stopped. Processing...")
    
    wav_path = save_to_wav(audio)
    
    try:
        user_input = await transcribe(wav_path)
        st.session_state.conversation.append(("You", user_input))
        
        reply = await generate_reply(user_input)
        st.session_state.conversation.append(("Bot", reply))
        
        await speak(reply)
        status.empty()
        
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        os.remove(wav_path)

# Display conversation
for speaker, text in st.session_state.conversation:
    if speaker == "You":
        st.markdown(f"**🗣️ You:** {text}")
    else:
        st.markdown(f"**🤖 Bot:** {text}")

# Record button
if st.button("Start Recording"):
    if not api_key:
        st.error("Please enter a valid OpenAI API key")
    else:
        asyncio.run(run_chat())

# Clear conversation button
if st.button("Clear Conversation"):
    st.session_state.conversation = []
    st.experimental_rerun()