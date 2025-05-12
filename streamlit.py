import streamlit as st
import asyncio
import os
import tempfile
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer

# Set Streamlit page configuration
st.set_page_config(page_title="🗣️ Hinglish Voice Chatbot", layout="centered")

# Title and description
st.title("🗣️ Hinglish Voice Chatbot")
st.write("Upload a `.wav` file (16kHz mono PCM), and the bot will transcribe and respond in Hinglish.")

# Input for OpenAI API Key
openai_key = st.text_input("Enter your OpenAI API key:", type="password")

# Check if API key is provided
if not openai_key:
    st.warning("Please enter your OpenAI API key to proceed.")
    st.stop()

# Initialize OpenAI client
client = AsyncOpenAI(api_key=openai_key)

# File uploader for audio files
uploaded_file = st.file_uploader("Upload your voice recording", type=["wav"])

# Function to transcribe audio using Whisper
async def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = await client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="text"
        )
    return transcript.strip()

# Function to generate a response using GPT-4
async def generate_response(prompt):
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a friendly Hinglish customer support agent."},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content.strip()

# Function to convert text to speech and play it
async def speak_text(text):
    async with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="onyx",
        input=text,
        response_format="pcm"
    ) as response:
        await LocalAudioPlayer().play(response)

# Process the uploaded audio file
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(uploaded_file.read())
        temp_audio_path = temp_audio_file.name

    # Display the uploaded audio file
    st.audio(uploaded_file, format="audio/wav")

    # Button to start processing
    if st.button("Transcribe and Respond"):
        with st.spinner("Processing..."):
            try:
                # Transcribe the audio
                transcription = await transcribe_audio(temp_audio_path)
                st.success(f"Transcription: {transcription}")

                # Generate a response
                response = await generate_response(transcription)
                st.info(f"🤖 Bot: {response}")

                # Speak the response
                await speak_text(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                # Clean up the temporary file
                os.remove(temp_audio_path)
