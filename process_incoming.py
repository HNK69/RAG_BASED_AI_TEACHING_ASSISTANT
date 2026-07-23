import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests 


def create_embedding(text_list):
        r = requests.post("http://localhost:11434/api/embed",json = {
            "model":"bge-m3",
            "input": text_list
        })
        # print(r.json())
        embedding = r.json()["embeddings"]
        return embedding

df = joblib.load('embeddings.joblib')


incoming_query = input("Ask A question: ")
question_embedding = create_embedding([incoming_query])[0]
# print(question_embedding)

#find similarities of question_embeddings withother embeddings.
# print(np.vstack(df["embedding"]).shape)

# print(np.vstack(df["embedding"]))     
similarities = cosine_similarity(np.vstack(df["embedding"]),[question_embedding]).flatten()
top_results = 10
# print(similarities)
max_index=(similarities.argsort()[::-1][0:top_results])
new_df = df.loc[max_index]
# print(new_df[["title","text"]])

for index,item in new_df.iterrows():
        print(index,item["title"],item["text"],item["start"],item["end"])
