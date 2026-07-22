import whisper
import json

model = whisper.load_model("small")

result = model.transcribe(audio='audios/sample.mp3',
                          language='en',
                          task='transcribe')
# print(result)
chunks = []
for segment in result['segments']:
    chunks.append({"start":segment['start'],"end":segment['end'],"text":segment['text']})

# print(chunks)
with open('output.json','w') as f:
    json.dump(chunks,f)