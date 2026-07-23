import requests
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
        r = requests.post("http://localhost:11434/api/embed",json = {
            "model":"bge-m3",
            "input": text_list
        })
        # print(r.json())
        embedding = r.json()["embeddings"]
        return embedding
# a = create_embedding("hi")
# print(a)

jsons = os.listdir("jsons") #list all jsons
# print(jsons)
my_dicts = []
chunk_id=0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    # print(content)
    print(f"creating embeddings for {json_file}")
    embeddings = create_embedding([c["text"] for c in content["chunks"]])
    # print(embeddings)
    for i,chunk in enumerate(content["chunks"]):
        # print(chunk)
        chunk["chunk_id"]=chunk_id
        chunk["embedding"]=embeddings[i]
        chunk_id+=1
        my_dicts.append(chunk)
    #     if (i==3):
    #          break
    # #     print(chunk)
    # break

# print(my_dicts)

# a = create_embedding(["hi","hello"])
# print(a)

df = pd.DataFrame.from_records(my_dicts)
# print(df)
joblib.dump(df,'embeddings.joblib')
