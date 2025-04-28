import os
import torch
import librosa
import speech_recognition as sr
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

def transcribe_audio_google(audio_file):
    """Transcribes audio using Google Web Speech API"""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)  # Read the audio file
    try:
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError:
        return "Could not request results from Google API."

def transcribe_audio_wav2vec(audio_file):
    """Transcribes audio using Wav2Vec2 (Offline)"""
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    
    # Load audio and convert to 16kHz
    audio, rate = librosa.load(audio_file, sr=16000)
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values
    
    with torch.no_grad():
        logits = model(input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]

def main():
    """Main function to choose method and transcribe audio"""
    audio_path = r"C:\Users\Sachin m\Downloads\sample.wav"# Change to your audio file path
    if not os.path.exists(audio_path):
        print("Error: Audio file not found!")
        return
    
    print("Choose transcription method:")
    print("1. Google API (Online)")
    print("2. Wav2Vec2 (Offline)")
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        result = transcribe_audio_google(audio_path)
    elif choice == "2":
        result = transcribe_audio_wav2vec(audio_path)
    else:
        print("Invalid choice!")
        return
    
    print("\nTranscription:")
    print(result)

if __name__ == "__main__":
    main()
