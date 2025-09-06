from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French.")
]

while True:
    user_input = input("Enter your message (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    messages.append(HumanMessage(content=user_input))
    
    model = ChatGoogleGenerativeAI(model = 'gemini-1.5-pro')
    response = model.invoke(messages)
    
    print("AI:", response.content)
    messages.append(AIMessage(content=response.content))

model = ChatGoogleGenerativeAI(model = 'gemini-1.5-pro')

