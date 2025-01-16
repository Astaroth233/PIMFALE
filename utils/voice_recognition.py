import librosa
import numpy as np
import base64
from scipy.spatial.distance import cosine

def capture_voice_data():
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Say something!")
        audio = recognizer.listen(source)

    with open('data/temp.wav', 'wb') as f:
        f.write(audio.get_wav_data())

    with open('data/temp.wav', 'rb') as f:
        voice_data = f.read()
    return base64.b64encode(voice_data).decode('utf-8')

def encode_voice(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

def calculate_voice_similarity(stored_voice_path, new_voice_data):
    new_voice_file = 'data/temp.wav'
    with open(new_voice_file, 'wb') as f:
        f.write(base64.b64decode(new_voice_data))

    stored_voice_encodings = encode_voice(stored_voice_path)
    new_voice_encodings = encode_voice(new_voice_file)

    if stored_voice_encodings is not None and new_voice_encodings is not None:
        similarity = 1 - cosine(stored_voice_encodings, new_voice_encodings)
        return similarity  # Similarity score between 0 and 1
    return 0
