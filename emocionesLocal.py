from tensorflow.keras.models import load_model
import librosa
import numpy as np
import os

CAT7 = ['fear', 'disgust', 'neutral', 'happy', 'sad', 'surprise', 'angry']

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

path = "test.wav"
mfccs_ = get_mfccs(path, model_.input_shape[-2])
mfccs_ = mfccs_.T.reshape(1, *mfccs_.T.shape)
pred_ = model_.predict(mfccs_)[0]
txt = "MFCCs\n" + get_title(pred_, CAT7)
print(CAT7[pred_.argmax()]) #Imprime emocion en pantalla


