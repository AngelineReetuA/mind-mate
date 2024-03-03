import pyttsx3
text_speech = pyttsx3.init()


def tts(text):
    return text_speech.say(text)
