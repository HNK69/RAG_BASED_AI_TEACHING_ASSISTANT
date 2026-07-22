import whisper
import json
import os

model = whisper.load_model("small")
audios = os.listdir("audios")
# print(audios)

for audio in audios:
    title = audio.split(".")[0]
    # print(title)
    result = model.transcribe(audio = f"audios/{audio}",
                              language="en",
                              task="transcribe",
                              word_timestamps=False )
    
    chunks = []
    for segment in result["segments"]:
        chunks.append({"title":title,"start":segment["start"],"end":segment["end"],"text":segment["text"]})

    chunks_with_metadata = {"chunks":chunks,"text":result["text"]}

    with open(f"jsons/{title}.json","w") as f:
        json.dump(chunks_with_metadata,f)