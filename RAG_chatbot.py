import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
db_path='chroma_db'

def get_rag_chain():

    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    db = Chroma(persist_directory=db_path,embedding_function=embedding_model)
    retriever = db.as_retriever(search_type='similarity',search_kwargs={'k':5})

    #connecting to llm
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    #prompt template

    template = '''you are an expert in structural engineering.
    only answer when asked related to structural engineering.
    dont makeup any formulas , if you dont know just say i dont know.
    any question asked should be answered in a detailed way with step by step explanation
    
    context:
    {context}
    
    question:
    {question}
    
    answer:'''

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = {'context':retriever| format_docs,'question':RunnablePassthrough()} | prompt | llm | StrOutputParser()
    return rag_chain


if __name__ == "__main__":
    chain = get_rag_chain()
    print("Chatbot Terminal Mode (Type 'exit' to quit)")
    while True:
        query = input("\nUser: ")
        if query.lower() == "exit":
            break
        response = chain.invoke(query)
        print(f"Bot: {response}")

