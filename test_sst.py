import speech.speech_to_text as stt
import speech.similary as sim

str = stt.speech_to_text()

results = sim.similary(str)
print(results)