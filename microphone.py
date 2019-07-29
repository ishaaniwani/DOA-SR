from pydub import AudioSegment

class Microphone(object):

    # Constructs new microphone object 
    # x - x coordinate in meters
    # y - y coordinate in meters 
    # fileName - path to recorded wav file. 
    def __init__(self, x, y, fileName):
        self.mx = x
        self.my = y
        self.mfileName = fileName
        self.sound = AudioSegment.from_wav(self.mfileName)
        self.sound.set_channels(1)
        self.durationSeconds = self.sound.duration_seconds

    # Returns x coordinate of microphone 
    def getX(self):
        return self.mx
    
    # Returns y coordinate of microphone
    def getY(self):
        return self.my

    # Returns fileName of microphone's recorded wav file 
    def getFileName(self):
        return self.mfileName

    # Returns AudioSegment representation of wav file 
    def getSound(self):
        return self.sound

    # Returns the duration in seconds of the wav file
    def getDurationSeconds(self):
        return self.durationSeconds