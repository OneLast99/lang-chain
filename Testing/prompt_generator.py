from langchain_core.prompts import PromptTemplate


template = PromptTemplate(template=""" Explain the research paper titled "{paper_input}". Explanation style : {style_input}. Explanation length : {length_input}. Use clear and concise language to ensure the explanation is easy to understand. If certain information is not available, respond with "Information not available" instead of making assumptions. Ensure it is alligned with the selected style and length. Make sure it doesn't exceed the length mentioned.
""",
    input_variables=["paper_input", "style_input", "length_input"],
    validate_template=True
)

template.save('template.json')