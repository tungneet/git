import streamlit as st
import numpy as np
import tempfile
import os
import asyncio
from io import BytesIO
from openai import AsyncOpenAI
from pydub import AudioSegment
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit app setup
st.title("🎤 Hinglish Voice Chatbot")
st.write("Speak to the bot in English or Hinglish and get a friendly response!")

# API Key Input
api_key = st.text_input("Enter your OpenAI API Key:", type="password", key="api_key")
if not api_key:
    st.warning("Please enter your OpenAI API key to continue")
    st.stop()

# Initialize client in session state
if 'client' not in st.session_state:
    st.session_state.client = AsyncOpenAI(api_key=api_key)

# Initialize conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Audio recording
def record_audio():
    try:
        audio_bytes = st.audio_input("Speak now:", key="audio_recorder")
        if audio_bytes:
            logger.info("Audio recorded successfully")
            audio = AudioSegment.from_file(BytesIO(audio_bytes), format="webm")
            wav_io = BytesIO()
            audio.export(wav_io, format="wav")
            return wav_io.getvalue()
        return None
    except Exception as e:
        logger.error(f"Recording error: {str(e)}")
        st.error(f"Recording error: {str(e)}")
        return None

# Async operations runner
async def run_async_operations(audio_bytes):
    try:
        # Transcribe
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp.seek(0)
            transcript = await st.session_state.client.audio.transcriptions.create(
                file=tmp,
                model="whisper-1",
                response_format="text"
            )
            logger.info(f"Transcription: {transcript}")
        
        # Generate response
        response = await st.session_state.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a friendly Hinglish customer support agent."},
                {"role": "user", "content": transcript}
            ],
        )
        reply = response.choices[0].message.content
        logger.info(f"Generated reply: {reply}")
        
        return transcript.strip(), reply.strip()
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise e

# Main chat function
def handle_chat():
    status = st.empty()
    status.info("🎤 Recording... Click the microphone and speak")
    
    audio_bytes = record_audio()
    
    if audio_bytes:
        status.info("🔄 Processing your request...")
        
        try:
            # Run async operations
            transcript, reply = asyncio.run(run_async_operations(audio_bytes))
            
            # Update conversation
            st.session_state.conversation.append(("You", transcript))
            st.session_state.conversation.append(("Bot", reply))
            
            status.empty()
            st.success("✅ Done!")
            
            # Auto-scroll to latest message
            st.experimental_rerun()
            
        except Exception as e:
            status.error(f"❌ Error: {str(e)}")

# Display conversation
chat_container = st.container()
with chat_container:
    for speaker, text in st.session_state.conversation:
        if speaker == "You":
            st.markdown(f"**🗣️ You:** {text}")
        else:
            st.markdown(f"**🤖 Bot:** {text}")

# Chat controls
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Voice Chat", key="start_chat"):
        handle_chat()
with col2:
    if st.button("Clear Conversation", key="clear_chat"):
        st.session_state.conversation = []
        st.experimental_rerun()

# Debug info (can be removed in production)
if st.checkbox("Show debug info"):
    st.write("Session state:", st.session_state)
    st.write("OpenAI client initialized:", 'client' in st.session_state)