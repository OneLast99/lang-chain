from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

# Force CPU device
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=-1  # <--- key change
)

llm = HuggingFacePipeline(pipeline=pipe)

model = ChatHuggingFace(llm=llm)

res = model.invoke("Why is India poor?")
print(res)