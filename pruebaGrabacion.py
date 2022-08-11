import sounddevice as sd 
from scipy.io.wavfile import write 
import wavio as wv 

def grabador():
    freq = 44100
    duration = 5
    recording = sd.rec(int(duration * freq),  
                   samplerate=freq, channels=2)   
    sd.wait() 
    write("grabacion.wav", freq, recording) 
    return()

grabador()
