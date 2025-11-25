import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

data_path = 'data'
db_path = 'chroma_db'
def main():

    #Loading Data
    documents = []
    for filename in os.listdir(data_path):

        if filename.endswith('.pdf'):
            try:
                file_path = os.path.join(data_path, filename)
                loading = PyPDFLoader(file_path)
                documents.extend(loading.load())
            except Exception as e:
                print(e)

    print(f"Total pages loaded: {len(documents)}")

    #Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    #Creating Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory= db_path
    )

    print('data ingestion complete')

if __name__ == "__main__":
    main()


