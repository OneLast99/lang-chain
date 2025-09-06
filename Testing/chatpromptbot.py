from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

chat_template = ChatPromptTemplate([
    ('system', "You are a helpful {domain} expert."),
    ('human', "{question}")
    ])

prmpt = chat_template.invoke({
    'domain':'math',
    'question':'What is the Riemann Hypothesis?'
})

print(prmpt)