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
def inference(prompt):
        r = requests.post("http://localhost:11434/api/generate",json = {
                "model" : "qwen3",
                "prompt" : prompt,
                "stream" : False,
                "think" : False,
                "options" : {
                        "temperature" : 0.2,    
                        "num_predict" : 512
                }
        })
        return r.json()["response"]

df = joblib.load('embeddings.joblib')


incoming_query = input("Ask A question: ")
question_embedding = create_embedding([incoming_query])[0]
# print(question_embedding)

#find similarities of question_embeddings withother embeddings.
# print(np.vstack(df["embedding"]).shape)

# print(np.vstack(df["embedding"]))     
similarities = cosine_similarity(np.vstack(df["embedding"]),[question_embedding]).flatten()
top_results = 8 
# print(similarities)
max_index=(similarities.argsort()[::-1][0:top_results])
new_df = df.loc[max_index]
# print(new_df[["title","text"]])
    
prompt = f'''
You are Teaching Assistant for the videos.
Here are video subtitle chunks containing video title , video number , start time in seconds , end time in seconds , the texts as that:

{new_df[["title","start","end","text"]].to_json()}
----------------------------------------------------------------------------------------------------------------
"{incoming_query}"

user asked this question related to the video chunks ,  you have to answer where and how much content is taught in which video (in which video and at what timestamp) and
guide the user to go to that particular video and you should never give back the chunks to user , you should only give better response than other models.
If user asks unrelated question , tell him that you can only answer questions related to the course. 
'''
with open("prompt.txt","w",encoding="utf-8") as f:
        f.write(prompt)

# for index,item in new_df.iterrows():
#         print(index,item["title"],item["text"],item["start"],item["end"])

response_by_llm = inference(prompt)
# print(response_by_llm)
with open("response.txt","w",encoding="utf-8") as f:
        f.write(response_by_llm)