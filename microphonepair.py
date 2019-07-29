import scipy.io.wavfile as wavfile
import speech_recognition as sr
from microphone import Microphone
from pydub import AudioSegment
from vad import VoiceActivityDetector
from xcorr import xcorr

class MicrophonePair(object):

    def getAmplitudeList(self, fileName):
        amplitudes = wavfile.read(fileName)[1]
        return amplitudes

    # Returns the start of speech, in ms, of a wav file. If no speech is detected, then returns -1
    def getSpeechStartTime(self, fileName):
        v = VoiceActivityDetector(fileName)
        windows = v.detect_speech()
        for i in range(0, len(windows)):
            arr = windows[i]
            if arr[len(arr) - 1] == 1:
                # VAD breaks wav file into 20 ms windowed chunks, with a 10 ms overlay
                return i * 10 
        return -1 
    
    # Returns the coefficents and delays of the cross correlation function of two arrays
    # a and b should be the same length
    def crossCorrelate(self, a, b, maxDelay):
        # maxDelay cannot be greater the length of the arrays, that will cause errors
        assert maxDelay < len(a)

        lags, coeffs = xcorr(a, b, maxlags=maxDelay)
        return lags, coeffs

    def __init__(self, mic1, mic2):
        # Check if there is speech, if not, then there is no point in doing 
        # further calculations. 
        speechStartTime = self.getSpeechStartTime(mic1.getFileName())
        if speechStartTime == -1:
            self.speechExists = False
            return 
        else: 
            self.speechExists = True

        # Get speech portion from both wav files    
        speech1 = mic1.getSound()[speechStartTime - 100:speechStartTime + 400]
        speech2 = mic2.getSound()[speechStartTime - 100:speechStartTime + 400]

        speech1Holder = 'holder/speech1.wav'
        speech2Holder = 'holder/speech2.wav'

        speech1.export(speech1Holder,format='wav')
        speech2.export(speech2Holder,format='wav')

        # Prepare an amplitudes list for the speech portions 
        # Needs to be normalized and values need to be lessened 
        # so that cross correlation does not deal with the square root 
        # of negative numbers and there is not an overflow error 
        data1 = self.getAmplitudeList(speech1Holder)
        data2 = self.getAmplitudeList(speech2Holder)

        self.speech1 = []
        self.speech2 = []

        for num in data1:
            self.speech1.append(abs(num / 1000))
        
        for num in data2:
            self.speech2.append(abs(num / 1000))
        
        # Get the mean amplitude of each speaking portion 
        meanSpeech1 = sum(self.speech1) / len(self.speech2)
        meanSpeech2 = sum(self.speech1) / len(self.speech2)

        # Recognize the speech of the wav files 
        # Give the wav file with more mean amplitude priority
        if meanSpeech1 > meanSpeech2:
            self.DOA = 45
            r = sr.Recognizer()
            with sr.WavFile(speech1Holder) as source:
                audio = r.record(source)
            text = r.recognize_google(audio)
            self.recognizedSpeech = text
        else:
            self.DOA = 135
            r = sr.Recognizer()
            with sr.WavFile(speech2Holder) as source:
                audio = r.record(source)
            text = r.recognize_google(audio)
            self.recognizedSpeech = text
    
    # Getter functions for microphone array class

    # Get the estimated direction of arrival of the sound source 
    def getDOA(self):
        return self.DOA

    # Get the recognized speech signal of the sound source
    def getRecognizedSpeech(self):
        return self.recognizedSpeech

    # Get the normalized and lessened amplitude values of the speech portion of the first wav file
    def getSpeech1(self):
        return self.speech1

    # Get the normalized and lessened amplitude values of the speech portion of the second wav file
    def getSpeech2(self):
        return self.speech2

    # Return whether or not speech actually exists inside of the wav file 
    # This will be used for the microphone array class. 
    def getSpeechExists(self):
        return self.speechExists       