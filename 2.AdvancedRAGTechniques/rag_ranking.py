from pypdf import PdfReader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import numpy as np
from sentence_transformers import CrossEncoder

reader = PdfReader('investor-presentation-q4fy25.pdf')
text = [p.extract_text().strip() for p in reader.pages]

len(text) # Number Of pages
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(text))
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

embedding_function = SentenceTransformerEmbeddingFunction()
chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(
    "lt-finance-summary-an", embedding_function=embedding_function
)
ids = [str(i) for i in range(len(token_split_texts))]

chroma_collection.add(ids=ids, documents=token_split_texts)
count = chroma_collection.count()
print(count)
q='Executive Summary'
results = chroma_collection.query(
    query_texts=q, n_results=10, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"][0]

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

pairs = [[q, doc] for doc in retrieved_documents]
scores = cross_encoder.predict(pairs)

print("Scores:")
for score in scores:
    print(score)

print("New Ordering:")
for o in np.argsort(scores)[::-1]:
    print(o + 1)


"""
Example Output
Scores:
-10.702773
-7.9567766
-10.705412
-11.311729
-10.29338
-11.256178
1.0214869
-10.721673
-11.305632
-11.269822
New Ordering:
7
2
5
1
3
8
6
10
9
4
"""