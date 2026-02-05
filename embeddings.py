import faiss
from sentence_transformers import SentenceTransformer
import numpy as np


model=SentenceTransformer("all-MiniLM-l6-v2")


sentences=[
    "I want to learn llm",
    "I want to build the LLM applications",
    "I will be learning langchain today",
    " I will do the embeddings today",
    "fix the python bug",
    "learn to cook"
]

embeddings=model.encode(sentences)

print(embeddings.shape)

def cosine_similarity(a,b):
    return np.dot(a,b)/np.linalg.norm(a)*np.linalg.norm(b)
print(cosine_similarity(embeddings[0], embeddings[1]))
print(cosine_similarity(embeddings[0], embeddings[2]))

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

print("Total vectors stored:", index.ntotal)



