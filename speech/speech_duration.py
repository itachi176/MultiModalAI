from mutagen.mp3 import MP3

def mutagen_length(path):
    try:
        audio = MP3(path)
        length = audio.info.length
        return length
    except:
        return None

# length = mutagen_length('../data/output.mp3')
# print("duration sec: " + str(length))
# print("duration min: " + str(int(length/60)) + ':' + str(int(length%60)))