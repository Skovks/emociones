from tensorflow.keras.models import load_model
import librosa
import numpy as np
import os

import sounddevice as sd 
from scipy.io.wavfile import write 
import wavio as wv 
CAT7 = ['fear', 'disgust', 'neutral', 'happy', 'sad', 'surprise', 'angry']

def grabador():
    freq = 44100
    duration = 5
    print("comienza grabacion")
    recording = sd.rec(int(duration * freq),  
                   samplerate=freq, channels=2)   
    sd.wait() 
    print("termina grabacion")
    write("grabacion.wav", freq, recording)

def get_title(predictions, categories):
    title = f"Detected emotion: {categories[predictions.argmax()]} \
    - {predictions.max() * 100:.2f}%"
    return title

def get_mfccs(audio, limit):
    y, sr = librosa.load(audio)
    a = librosa.feature.mfcc(y, sr=sr, n_mfcc=40)
    if a.shape[1] > limit:
        mfccs = a[:, :limit]
    elif a.shape[1] < limit:
        mfccs = np.zeros((a.shape[0], limit))
        mfccs[:, :a.shape[1]] = a
    return mfccs

model_ = load_model("model4.h5")
while(True):
    grabador()
    path ="grabacion.wav"
    mfccs_ = get_mfccs(path, model_.input_shape[-2])
    mfccs_ = mfccs_.T.reshape(1, *mfccs_.T.shape)
    pred_ = model_.predict(mfccs_)[0]
    txt = "MFCCs\n" + get_title(pred_, CAT7)
    print(CAT7[pred_.argmax()]) #Imprime emocion en pantalla


