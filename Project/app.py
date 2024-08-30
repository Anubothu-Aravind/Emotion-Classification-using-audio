import streamlit as st

def add_transcription_to_history(source, text, emotions):
    timestamp = time.strftime('%B %d, %Y at %I:%M')
    emotion_analysis = "\n".join([f"{emotion.capitalize()}: {score:.2%}" for emotion, score in emotions.items()])
    dominant_emotion = max(emotions, key=emotions.get)
    entry = f'Audio Received at "{timestamp}"\n\nAudio Loaded From "{source}"\n\nTranscribed Text:\n\n{text[:100]}...\n\nEmotion Analysis:\n\n{emotion_analysis}\n\nBossy emotion : {dominant_emotion.capitalize()} ({emotions[dominant_emotion]:.2%})'
    st.session_state['history'].insert(0, entry)


st.title("Emotion Classification App")

input_method = st.selectbox("Choose audio input method:", ["Record Audio", "Upload File"])

if 'history' not in st.session_state:
    st.session_state['history'] = []

st.sidebar.header("Transcription History")

if st.session_state['history']:
    for i, entry in enumerate(reversed(st.session_state['history']), 1):
        with st.sidebar.expander(f"Transcription {i}", expanded=(i == 1)):
            st.write(entry)
    if st.sidebar.button("Clear History"):
        st.session_state['history'] = []
        st.rerun()
else:
    st.sidebar.write("No Previous Transcription Available.")

