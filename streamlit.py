import streamlit as st
import numpy as np
import scipy.io.wavfile as wavfile
import tempfile
import os
from openai import AsyncOpenAI
from streamlit_microphone import streamlit_microphone

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

# Audio recording using browser microphone
def record_audio():
    audio_bytes = streamlit_microphone(
        start_prompt="🎤 Start speaking",
        stop_prompt="⏹️ Stop recording",
        just_once=True,
        use_container_width=True
    )
    return audio_bytes

# Save audio bytes to WAV file
def save_to_wav(audio_bytes):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    
    # Convert bytes to numpy array (assuming 16-bit mono at 16kHz)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    
    # Write to WAV file
    wavfile.write(temp_file.name, 16000, audio_np)
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

# Main chat function
async def run_chat():
    status = st.empty()
    status.info("🎤 Recording... Speak now")
    
    audio_bytes = record_audio()
    
    if audio_bytes:
        status.success("🔇 Recording stopped. Processing...")
        wav_path = save_to_wav(audio_bytes)
        
        try:
            user_input = await transcribe(wav_path)
            st.session_state.conversation.append(("You", user_input))
            
            reply = await generate_reply(user_input)
            st.session_state.conversation.append(("Bot", reply))
            
            st.markdown(f"**🤖 Bot:** {reply}")
            status.empty()
            
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

# Display conversation
for speaker, text in st.session_state.conversation:
    if speaker == "You":
        st.markdown(f"**🗣️ You:** {text}")
    else:
        st.markdown(f"**🤖 Bot:** {text}")

# Start chat button
if st.button("Start Voice Chat"):
    asyncio.run(run_chat())

# Clear conversation button
if st.button("Clear Conversation"):
    st.session_state.conversation = []
    st.experimental_rerun()