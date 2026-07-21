#convert the video to mp3
import subprocess
import os
files = os.listdir('videos')
# print(files)
for file in files:
    subprocess.run(["ffmpeg","-i",f"videos/{file}",f"audios/{1}.mp3"])