from gtts import gTTS
import playsound
def text_to_speech(text):
    output = gTTS(text,lang="vi", slow=False)
    output.save("output.mp3")
    playsound.playsound('output.mp3', True)
