import asyncio
import streamlit as st
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer
import tempfile
import os
import numpy as np
import scipy.io.wavfile as wavfile
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

# Streamlit UI for OpenAI Key input
openai_key = st.text_input("Enter your OpenAI API key:", type="password")

if not openai_key:
    st.warning("Please enter an OpenAI API key to proceed.")
else:
    client = AsyncOpenAI(api_key=openai_key)

    SAMPLE_RATE = 16000
    THRESHOLD = 1000
    SILENCE_DURATION = 1.5

    # Define the WebRTC Audio Processor
    class AudioProcessor(AudioProcessorBase):
        def recv(self, frame):
            # Process the audio frames
            audio_data = np.frombuffer(frame.to_bytes(), dtype=np.int16)
            return audio_data

    # Record audio using WebRTC
    def record_audio():
        audio_processor = AudioProcessor()
        webrtc_streamer(
            key="audio_input",
            mode=WebRtcMode.SENDRECV,
            audio_processor_factory=lambda: audio_processor,
        )

        return audio_processor.recv

    # Save to WAV file
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

    # Generate reply
    async def generate_reply(prompt):
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a friendly Hinglish customer support agent."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content.strip()

    # Speak
    async def speak(text):
        async with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="onyx",
            input=text,
            instructions="Speak in Indian accent in Hinglish. Use natural, friendly tone.",
            response_format="pcm"
        ) as response:
            await LocalAudioPlayer().play(response)

    # Streamlit UI
    def main():
        st.set_page_config(page_title="🗣️ Hinglish Voice Chatbot", layout="centered")
        st.title("🗣️ Hinglish Voice Chatbot")
        st.write("Click the button and start talking. It will listen, respond, and speak back in Hinglish.")

        if st.button("🎤 Start Talking"):
            with st.spinner("Recording... Speak now!"):
                audio_data = record_audio()
                wav_path = save_to_wav(audio_data)

            async def process_audio():
                try:
                    user_input = await transcribe(wav_path)
                    st.success(f"You said: {user_input}")

                    reply = await generate_reply(user_input)
                    st.info(f"🤖 Bot: {reply}")

                    await speak(reply)
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    os.remove(wav_path)

            asyncio.run(process_audio())

    if __name__ == "__main__":
        main()
