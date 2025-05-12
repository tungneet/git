import streamlit as st
import numpy as np
import tempfile
import os
import asyncio
from io import BytesIO
from openai import AsyncOpenAI
from pydub import AudioSegment

# Streamlit app setup
st.title("🎤 Hinglish Voice Chatbot")
st.write("Speak to the bot in English or Hinglish and get a friendly response!")

# API Key Input
api_key = st.text_input("Enter your OpenAI API Key:", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API key to continue")
    st.stop()

client = AsyncOpenAI(api_key=api_key)

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Audio recording using Streamlit's native audio recorder
def record_audio():
    audio_bytes = st.audio_input("Speak now:", key="audio_recorder")
    if audio_bytes:
        # Convert webm to wav using pydub
        audio = AudioSegment.from_file(BytesIO(audio_bytes), format="webm")
        wav_io = BytesIO()
        audio.export(wav_io, format="wav")
        return wav_io.getvalue()
    return None

# Transcribe using Whisper
async def transcribe(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.seek(0)
        transcript = await client.audio.transcriptions.create(
            file=tmp,
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

# Async function runner for Streamlit
def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# Main chat function
def run_chat():
    status = st.empty()
    status.info("🎤 Recording... Click the microphone and speak")
    
    audio_bytes = record_audio()
    
    if audio_bytes:
        status.success("🔇 Recording received. Processing...")
        
        try:
            user_input = run_async(transcribe(audio_bytes))
            st.session_state.conversation.append(("You", user_input))
            
            reply = run_async(generate_reply(user_input))
            st.session_state.conversation.append(("Bot", reply))
            
            st.markdown(f"**🤖 Bot:** {reply}")
            status.empty()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Display conversation
for speaker, text in st.session_state.conversation:
    if speaker == "You":
        st.markdown(f"**🗣️ You:** {text}")
    else:
        st.markdown(f"**🤖 Bot:** {text}")

# Start chat button
if st.button("Start Voice Chat"):
    run_chat()

# Clear conversation button
if st.button("Clear Conversation"):
    st.session_state.conversation = []
    st.experimental_rerun()