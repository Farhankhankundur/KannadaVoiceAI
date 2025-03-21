import gradio as gr
from deep_translator import GoogleTranslator
from TTS.api import TTS
import speech_recognition as sr
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import wiener, savgol_filter
import soundfile as sf
from resemblyzer import VoiceEncoder
import logging
from transformers import pipeline
import json
import os

# Initialize Components
device = "cpu"
vits_model = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", gpu=False)
voice_encoder = VoiceEncoder()

# Load the existing dictionary from a file
local_dict_path = "local_kannada_dictionary.json"
if os.path.exists(local_dict_path):
    with open(local_dict_path, 'r') as f:
        kannada_dictionary = json.load(f)
else:
    raise FileNotFoundError(f"Dictionary file not found at {local_dict_path}")

# Initialize emotion detection model
emotion_classifier = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', return_all_scores=True)

def detect_context(text):
    """
    Automatically detect if the context is related to drinking or food from both Kannada and English text
    """
    # For simplicity, we'll use a generic context detection for now
    return "generic"

def check_word_in_dictionary(word, context='generic'):
    """
    Check if a Kannada word exists in dictionary and return context-based translation.
    """
    if word in kannada_dictionary:
        meanings = kannada_dictionary[word]
        # Return the context-specific translation if available, else return the first meaning
        return meanings.get(context, next(iter(meanings.values()), None))
    return None

def translate_text(text, context=None):
    """
    Translate text with context awareness for specific Kannada words.
    Prioritizes the local dictionary before using Google Translator.
    """
    try:
        if context is None:
            context = "generic"
        
        # Split text into sentences (preserve punctuation)
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        translated_sentences = []
        
        for sentence in sentences:
            # Split sentence into words (preserve punctuation)
            words = sentence.split()
            translated_words = []
            
            # First, replace words with dictionary translations if available
            for word in words:
                clean_word = word.strip(".,!?")
                dictionary_translation = check_word_in_dictionary(clean_word, context)
                if dictionary_translation:
                    # If the word has a translation, use it
                    translated_words.append(dictionary_translation)
                else:
                    # Otherwise, keep the original word for now
                    translated_words.append(word)
            
            # Reconstruct the sentence with dictionary translations
            preprocessed_sentence = " ".join(translated_words)
            
            # Translate the preprocessed sentence using Google Translator
            translator = GoogleTranslator(source='kn', target='en')
            translated_sentence = translator.translate(preprocessed_sentence)
            
            translated_sentences.append(translated_sentence)
        
        # Reconstruct the full text with translated sentences
        return ". ".join(translated_sentences)
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        return "Translation error occurred"

def detect_emotion(text):
    """
    Detect emotion from the input text
    """
    try:
        results = emotion_classifier(text)
        emotions = results[0]
        max_emotion = max(emotions, key=lambda x: x['score'])
        return max_emotion['label']
    except Exception as e:
        logging.error(f"Emotion detection error: {str(e)}")
        return "neutral"

def generate_waveform(audio_path, after_denoising=False):
    """
    Generate and return a waveform plot from an audio file.
    Handles both before and after noise reduction.
    """
    y, sr = librosa.load(audio_path, sr=None)
    if after_denoising:
        y = librosa.effects.preemphasis(y)
        y = wiener(y)
        y = librosa.util.normalize(y)
    waveform_path = "after_waveform.png" if after_denoising else "before_waveform.png"
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, color="blue" if after_denoising else "green")
    plt.title("Waveform (After Denoising)" if after_denoising else "Waveform (Before Denoising)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(waveform_path)
    plt.close()
    return waveform_path

def smooth_audio(audio_path):
    """
    Apply a smoothing filter to the audio to reduce harsh transitions.
    """
    y, sr = librosa.load(audio_path, sr=None)
    y_smooth = savgol_filter(y, window_length=101, polyorder=2)
    smoothed_audio_path = "smoothed_audio.wav"
    sf.write(smoothed_audio_path, y_smooth, sr)
    return smoothed_audio_path

def calculate_voice_cloning_accuracy(input_audio, cloned_audio):
    """
    Calculate the similarity between the input voice and the cloned voice embeddings.
    """
    y1, sr1 = librosa.load(input_audio, sr=16000, duration=10.0)
    y2, sr2 = librosa.load(cloned_audio, sr=16000, duration=10.0)
    min_len = min(len(y1), len(y2))
    y1, y2 = y1[:min_len], y2[:min_len]
    input_embedding = voice_encoder.embed_utterance(y1)
    cloned_embedding = voice_encoder.embed_utterance(y2)
    similarity = np.dot(input_embedding, cloned_embedding) / (np.linalg.norm(input_embedding) * np.linalg.norm(cloned_embedding))
    accuracy = similarity * 100
    return f"{accuracy:.2f}%"

def process_audio(audio_path):
    """
    Function to process the audio file:
    - Convert speech to text (ASR)
    - Translate text (Kannada to English)
    - Perform Voice Cloning (VITS)
    - Generate waveforms (before and after noise reduction)
    - Calculate embedding accuracy for voice cloning
    """
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        transcription = recognizer.recognize_google(audio_data, language="kn-IN")
        translated_text = translate_text(transcription)
        emotion = detect_emotion(translated_text)
        cloned_audio_path = "cloned_audio.wav"

        # Adjust TTS parameters based on detected emotion
        if emotion == "happy":
            speed = 0.8  # Slightly faster for happy emotion
            pitch = 0.1  # Higher pitch for happy emotion
            emphasis = 1.2  # More emphasis for happy emotion
        elif emotion == "sad":
            speed = 0.6  # Slower for sad emotion
            pitch = -0.2  # Lower pitch for sad emotion
            emphasis = 1.0  # Less emphasis for sad emotion
        else:
            speed = 0.7  # Default speed for neutral emotion
            pitch = 0.0  # Default pitch for neutral emotion
            emphasis = 1.1  # Default emphasis for neutral emotion

        # Generate cloned voice with adjusted parameters
        vits_model.tts_to_file(
            translated_text, 
            speaker_wav=audio_path, 
            file_path=cloned_audio_path, 
            language="en",
            speed=speed, 
            pitch=pitch, 
            emphasis=emphasis, 
            noise_scale=0.5,  # Reduced noise scale for smoother audio
            length_scale=1.2  # Increased length scale for slower speech
        )

        # Apply smoothing to the cloned audio
        smoothed_audio_path = smooth_audio(cloned_audio_path)

        # Generate waveforms
        before_waveform_path = generate_waveform(audio_path)
        after_waveform_path = generate_waveform(audio_path, after_denoising=True)

        # Calculate cloning voice accuracy
        voice_cloning_accuracy = calculate_voice_cloning_accuracy(audio_path, smoothed_audio_path)

        return (
            transcription,
            translated_text,
            before_waveform_path,
            after_waveform_path,
            smoothed_audio_path,
            voice_cloning_accuracy,
            emotion
        )
    except Exception as e:
        return f"Error occurred: {str(e)}", None, None, None, None, None, None

def add_word_to_dictionary(kannada_word, english_meaning, context):
    """
    Add a new word to the local dictionary
    """
    global kannada_dictionary
    if kannada_word not in kannada_dictionary:
        kannada_dictionary[kannada_word] = {}
    kannada_dictionary[kannada_word][context] = english_meaning
    with open(local_dict_path, 'w') as f:
        json.dump(kannada_dictionary, f)
    return f"Added '{kannada_word}' with meaning '{english_meaning}' in context '{context}' to the dictionary."

def show_dictionary():
    """
    Display the current contents of the dictionary
    """
    dict_str = ""
    for word, meanings in kannada_dictionary.items():
        for context, meaning in meanings.items():
            dict_str += f"{word} ({context}): {meaning}\n"
    return dict_str

# Gradio Interface
def build_interface():
    with gr.Blocks(css="""
        :root {
            --primary: #6366f1;
            --secondary: #4f46e5;
            --accent: #a5b4fc;
            --background: #0f172a;
            --text: #e2e8f0;
        }
        .container {
            background: linear-gradient(45deg, var(--background) 0%, #1e293b 100%);
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        .gradio-button {
            background: var(--primary) !important;
            color: white !important;
            padding: 12px 24px !important;
            border-radius: 999px !important;
            transition: all 0.3s ease !important;
            border: none !important;
        }
        .gradio-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px var(--accent);
        }
        .input-section {
            background: rgba(30, 41, 59, 0.8);
            padding: 2rem;
            border-radius: 1rem;
            border: 1px solid var(--accent);
        }
        .output-card {
            background: rgba(15, 23, 42, 0.9);
            padding: 1.5rem;
            border-radius: 1rem;
            margin: 1rem 0;
            border: 1px solid var(--primary);
        }
        h1 {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(45deg, #818cf8, #a5b4fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem !important;
        }
        .waveform-img {
            border-radius: 12px;
            border: 2px solid var(--accent);
            padding: 8px;
            background: #1e293b;
        }
        .accuracy-badge {
            background: var(--primary);
            padding: 8px 16px;
            border-radius: 0; /* Remove rounded corners for square shape */
            font-weight: bold;
            display: inline-block;
            margin-top: 1rem;
            text-align: center; /* Center the text */
            width: 100%; /* Ensure the badge takes full width */
        }
        @keyframes gradient-pulse {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .gradio-interface {
            background: linear-gradient(45deg, #0f172a, #1e293b);
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        .gradio-tabs {
            background: rgba(15, 23, 42, 0.9);
            border-radius: 1rem;
            border: 1px solid var(--primary);
        }
        .gradio-tab-item {
            background: rgba(30, 41, 59, 0.8);
            border-radius: 1rem;
            border: 1px solid var(--accent);
        }
        /* Center the label text */
        .gradio-label {
            text-align: center !important;
            width: 100% !important;
            display: block !important;
        }
    """) as demo:
        
        gr.Markdown("""
        <h1 style="text-align: center; margin-bottom: 0;">
            <span style="display: inline-block; animation: float 3s ease-in-out infinite;">
            🚀 ಕನ್ನadaVoiceAI
            </span>
        </h1>
        <div style="text-align: center; margin-bottom: 2rem; color: #94a3b8;">
            Next-gen voice translation with neural cloning precision and emotion 
        </div>
        """)
        
        with gr.Row(variant="panel"):
            with gr.Column(scale=1, min_width=400):
                with gr.Group(elem_classes="input-section"):
                    gr.Markdown("### 🎤 Record Your Voice")
                    audio_input = gr.Audio(
                        type="filepath", 
                        label="Kannada Speech Input",
                        elem_classes="custom-audio",
                        waveform_options={"waveform_progress_color": "#818cf8"}
                    )
                    gr.Markdown("""
                    <div style="margin-top: 1rem; color: #94a3b8; font-size: 0.9em;">
                        Tip: Record in a quiet environment for best results
                    </div>
                    """)
                    
                    process_button = gr.Button(
                        "✨ Process & Transform Voice",
                        elem_id="magic-button"
                    )

            with gr.Column(scale=2, min_width=800):
                with gr.Tabs():
                    with gr.TabItem("📝 Transcription & Translation"):
                        with gr.Column(elem_classes="output-card"):
                            gr.Markdown("### 🔍 Transcription Analysis")
                            transcribed_text = gr.Textbox(
                                label="Original Kannada Text",
                                lines=3,
                                interactive=False,
                                show_copy_button=True
                            )
                            output_text = gr.Textbox(
                                label="Translated English Text",
                                lines=3,
                                interactive=False,
                                show_copy_button=True
                            )

                    with gr.TabItem("📊 Audio Visualization"):
                        with gr.Row():
                            with gr.Column(elem_classes="output-card"):
                                gr.Markdown("### 🌊 Waveform Analysis")
                                with gr.Row():
                                    before_waveform_output = gr.Image(
                                        label="Original Audio",
                                        elem_classes="waveform-img"
                                    )
                                    after_waveform_output = gr.Image(
                                        label="Enhanced Audio",
                                        elem_classes="waveform-img"
                                    )
                                gr.Markdown("""
                                <div style="text-align: center; margin-top: 1rem;">
                                    <span style="color: #818cf8;">●</span> Original 
                                    <span style="margin-left: 2rem; color: #a5b4fc;">●</span> Enhanced
                                </div>
                                """)

                    with gr.TabItem("🎙️ Voice Clone"):
                        with gr.Column(elem_classes="output-card"):
                            gr.Markdown("### 🔮 Cloned Voice Output")
                            output_audio = gr.Audio(
                                label="Synthesized English Speech",
                                interactive=False,
                                waveform_options={"waveform_progress_color": "#a5b4fc"}
                            )
                            voice_cloning_accuracy = gr.Label(
                                label="Neural Similarity Score",
                                value="Calculating...",
                                num_top_classes=1,
                                elem_classes=["accuracy-badge"]
                            )
                            detected_emotion = gr.Label(
                                label="Detected Emotion",
                                value="neutral",
                                num_top_classes=1,
                                elem_classes=["accuracy-badge"]
                            )
                            gr.Markdown("""
                            <div style="margin-top: 1rem; color: #94a3b8; font-size: 0.9em;">
                                Voice cloning accuracy measured using deep neural embeddings
                            </div>
                            """)

        process_button.click(
            process_audio, 
            inputs=audio_input, 
            outputs=[
                transcribed_text,
                output_text,
                before_waveform_output,
                after_waveform_output,
                output_audio,
                voice_cloning_accuracy,
                detected_emotion
            ]
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Add Word to Dictionary")
                kannada_word = gr.Textbox(label="Kannada Word")
                english_meaning = gr.Textbox(label="English Meaning")
                context = gr.Textbox(label="Context")
                add_word_button = gr.Button("Add Word")
                add_word_result = gr.Textbox(label="Result", interactive=False)

        add_word_button.click(
            add_word_to_dictionary,
            inputs=[kannada_word, english_meaning, context],
            outputs=add_word_result
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Current Dictionary")
                show_dict_button = gr.Button("Show Dictionary")
                dictionary_display = gr.Textbox(label="Dictionary Contents", interactive=False)

        show_dict_button.click(
            show_dictionary,
            outputs=dictionary_display
        )

    return demo

# Launch Interface
demo = build_interface()
demo.launch(share=True)