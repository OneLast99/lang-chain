from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership. He is the GOAT in cricket",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Virat Kohli is an Pakistani footballer knowna as a prolific goal scorer.",
    "Cristano Ronaldo is a Portuguese footballer who is the GOAT",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers.",
    "India's capital is New Delhi, a city rich in history and culture."
]

query = "Who is the best indian player?"

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

doc_embeddings = embedding.embed_documents(documents)

query_embedding = embedding.embed_query(query)

ans = cosine_similarity([query_embedding], doc_embeddings)[0]

print(sorted(list(enumerate(ans)), key=lambda x: x[1], reverse=True))