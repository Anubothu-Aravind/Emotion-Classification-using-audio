# Emotion Classification App

This application is a Streamlit-based tool for analyzing emotions from audio input. It utilizes the Hugging Face `transformers` library with the `j-hartmann/emotion-english-distilroberta-base` model for emotion detection from transcribed text. Users can either record audio or upload an audio file, which will then be transcribed and analyzed for emotional content. 

The app also displays the emotion analysis results in the form of both bar charts and pie charts.

## Features

- **Record Audio**: Users can record a short audio snippet (up to 30 seconds) directly in the app.
- **Upload Audio**: Users can upload audio files in WAV, MP3, OGG, FLAC, or AAC formats for transcription and emotion analysis.
- **Transcription**: Converts speech to text using the `speech_recognition` library with Google’s speech recognition service.
- **Emotion Analysis**: Utilizes the Hugging Face model `j-hartmann/emotion-english-distilroberta-base` to detect the emotions in the transcribed text.
- **Visualization**: Displays emotion probabilities as both a bar chart and a pie chart for easier analysis.
- **History**: Maintains a history of transcriptions and emotion analyses for reference.

## Installation

1. Clone the repository:
    ```bash
    git clone[https://github.com/your-username/emotion-classification-app.git
    ```

2. Navigate into the project directory:
    ```bash
    cd emotion-classification-app
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the root directory and add your Hugging Face token:
    ```bash
    HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
    ```

5. Run the application:
    ```bash
    streamlit run app.py
    ```

## Usage

1. **Record Audio**: Select "Record Audio", set the recording duration (up to 30 seconds), and click "Start Recording". After recording, the audio will be automatically processed and the transcription will be displayed alongside the emotion analysis.
   
2. **Upload Audio**: Select "Upload File", choose an audio file from your device, and the app will process the audio, display the transcription, and analyze the emotions.
   
3. **View History**: A sidebar shows previous transcriptions and their emotion analysis. You can clear the history by clicking the "Clear History" button.

## Emotion Analysis

The app uses a model trained on seven basic emotions:

- Anger
- Disgust
- Fear
- Joy
- Neutral
- Sadness
- Surprise

The emotions are displayed as probabilities, with a bar chart and pie chart providing a visual representation.

## Dependencies

- Python 3.9+
- Streamlit
- Hugging Face Transformers
- SpeechRecognition
- SoundDevice
- SoundFile
- Matplotlib
- Seaborn
- NumPy
- Torch

For a full list of dependencies, refer to `requirements.txt`.

## Notes

- This app requires an internet connection to use Google's speech recognition service and Hugging Face’s API for emotion classification.
- The app stores transcription and emotion analysis history during the session, which can be cleared manually from the sidebar.
