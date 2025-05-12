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

# Initialize session state
if 'client' not in st.session_state:
    api_key = st.text_input("Enter your OpenAI API Key:", type="password", key="api_key")
    if api_key:
        st.session_state.client = AsyncOpenAI(api_key=api_key)
    else:
        st.warning("Please enter your OpenAI API key to continue")
        st.stop()

if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Audio recording
def record_audio():
    try:
        audio_bytes = st.audio_input("Speak now:", key="audio_recorder")
        if audio_bytes and not st.session_state.processing:
            logger.info("Audio recorded - starting processing")
            st.session_state.processing = True
            audio = AudioSegment.from_file(BytesIO(audio_bytes), format="webm")
            wav_io = BytesIO()
            audio.export(wav_io, format="wav")
            return wav_io.getvalue()
        return None
    except Exception as e:
        logger.error(f"Recording error: {str(e)}")
        st.error(f"Recording error: {str(e)}")
        return None

# Process chat
async def process_chat(audio_bytes):
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp.seek(0)
            
            # Transcribe
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
        logger.error(f"Processing error: {str(e)}")
        raise e
    finally:
        st.session_state.processing = False

# Main chat handler
def handle_chat():
    if 'audio_recorder' in st.session_state and st.session_state.audio_recorder and not st.session_state.processing:
        with st.spinner("Processing your voice message..."):
            try:
                audio_bytes = st.session_state.audio_recorder['bytes']
                transcript, reply = asyncio.run(process_chat(audio_bytes))
                
                st.session_state.conversation.append(("You", transcript))
                st.session_state.conversation.append(("Bot", reply))
                
                # Clear the audio recorder to allow new recordings
                del st.session_state.audio_recorder
                st.experimental_rerun()
                
            except Exception as e:
                st.error(f"Error processing your request: {str(e)}")
                st.session_state.processing = False

# Display conversation
for speaker, text in st.session_state.conversation:
    if speaker == "You":
        st.markdown(f"**🗣️ You:** {text}")
    else:
        st.markdown(f"**🤖 Bot:** {text}")

# Automatic processing when new audio is recorded
handle_chat()

# Clear conversation button
if st.button("Clear Conversation"):
    st.session_state.conversation = []
    if 'audio_recorder' in st.session_state:
        del st.session_state.audio_recorder
    st.experimental_rerun()