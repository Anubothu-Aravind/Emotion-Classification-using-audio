import os
import warnings
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import time
import speech_recognition as sr
import soundfile as sf
import wave
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize session state for history if not already done
if 'history' not in st.session_state:
    st.session_state['history'] = []

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
hf_token = os.getenv("hf_token") or "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"  # Replace with your token if needed
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token
warnings.filterwarnings("ignore", category=FutureWarning, message=".*`clean_up_tokenization_spaces`.*")

@st.cache_resource
def load_model_and_tokenizer():
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

def convert_to_wav(input_path, output_path):
    try:
        data, samplerate = sf.read(input_path)
        sf.write(output_path, data, samplerate)
        return True
    except Exception as e:
        st.write(f"Error converting audio to WAV: {str(e)}")
        return False

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text
    except sr.UnknownValueError:
        return "Sorry, could not understand the audio."
    except sr.RequestError:
        return "Sorry, there was an error with the speech recognition service."

def analyze_emotions(text):
    tokenized_texts = tokenizer([text], truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokenized_texts)
    logits = outputs.logits.numpy()[0]
    probs = np.exp(logits) / np.exp(logits).sum()

    emotions = {
        'anger': probs[0],
        'disgust': probs[1],
        'fear': probs[2],
        'joy': probs[3],
        'neutral': probs[4],
        'sadness': probs[5],
        'surprise': probs[6],
    }

    return emotions

def plot_emotions(emotions):
    emotion_colors = {
        'anger': '#FF6347',
        'disgust': '#32CD32',
        'fear': '#1E90FF',
        'joy': '#FFD700',
        'neutral': '#262730',
        'sadness': '#4682B4',
        'surprise': '#FF69B4'
    }

    labels = list(emotions.keys())
    probabilities = list(emotions.values())

    fig, ax = plt.subplots(1, 2, figsize=(16, 10))
    sns.barplot(x=probabilities, y=labels, ax=ax[0], palette=emotion_colors)
    ax[0].set_title('Emotion Probabilities (Bar Chart)')
    ax[0].set_xlabel('Probability')

    wedges, texts, autotexts = ax[1].pie(
        probabilities, labels=labels, autopct='%1.1f%%',
        startangle=90, colors=[emotion_colors[label] for label in labels]
    )
    ax[1].set_title('Emotion Distribution (Pie Chart)')

    st.pyplot(fig)

def add_to_history(source, text, emotions):
    timestamp = time.strftime('%B %d, %Y at %I:%M %p')
    emotion_analysis = "\n".join([f"{emotion.capitalize()}: {score:.2%}" for emotion, score in emotions.items()])
    dominant_emotion = max(emotions, key=emotions.get)
    entry = f'Audio Received at "{timestamp}"\n\nAudio Loaded From "{source}"\n\nTranscribed Text:\n\n{text[:100]}...\n\nEmotion Analysis:\n\n{emotion_analysis}\n\nDominant emotion: {dominant_emotion.capitalize()} ({emotions[dominant_emotion]:.2%})'
    st.session_state['history'].insert(0, entry)

def process_audio(file_path, source):
    with st.spinner("Transcribing..."):
        text = transcribe_audio(file_path)

    st.write("--------------------------")
    st.write(f"Transcribed Text: {text}")

    with st.spinner("Analyzing emotions..."):
        emotions = analyze_emotions(text)

    plot_emotions(emotions)
    add_to_history(source, text, emotions)

st.title("Emotion Classification App")

st.sidebar.header("Transcription History")
if st.session_state['history']:
    for i, entry in enumerate(st.session_state['history'], 1):
        with st.sidebar.expander(f"Transcription {i}", expanded=(i == 1)):
            st.write(entry)
    if st.sidebar.button("Clear History"):
        st.session_state['history'] = []
        st.rerun()
else:
    st.sidebar.write("No Previous Transcription Available.")

input_method = st.selectbox("Choose audio input method:", ["Upload File"])

if input_method == "Upload File":
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac", "aac"])
    if uploaded_file is not None:
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            file_extension = uploaded_file.name.split('.')[-1].lower()
            temp_file_path = f"uploaded_audio_{timestamp}.{file_extension}"

            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if file_extension == 'wav':
                wav_file_path = temp_file_path
            else:
                wav_file_path = f"uploaded_audio_{timestamp}.wav"
                if not convert_to_wav(temp_file_path, wav_file_path):
                    raise Exception("Conversion to WAV failed")

            st.write("Processing audio file...")
            process_audio(wav_file_path, "Uploaded File")

            if os.path.exists(temp_file_path) and temp_file_path != wav_file_path:
                os.remove(temp_file_path)
            if os.path.exists(wav_file_path):
                os.remove(wav_file_path)

        except Exception as e:
            st.write(f"An error occurred: {str(e)}")
            st.write(f"Error type: {type(e).__name__}")
