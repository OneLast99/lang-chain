#from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
#from transformers import pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv

# Force CPU device
'''
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=-1  # <--- key change
)   
'''

#st.header('Research Tool')

#user_input = st.text_input("Enter your query:")

#if st.button('Submit'):
#    res = model.invoke(user_input)
#    st.write(res.content)

'''
llm = HuggingFacePipeline(pipeline=pipe)
model = ChatHuggingFace(llm=llm)
'''

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-1.5-pro')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1 paragraphs, 5 lines)", "Medium (2 paragraphs, each of 5 lines)", "Long (3 paragraphs, each of 6 lines)"] )

#template
template = load_prompt('template.json')



if st.button('Submit'):
    chain = template | model
    res = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })
    st.write(res.content)