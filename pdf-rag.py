## 1. Ingest PDF Files
# 2. Extract Text from PDF Files and split into small chunks
# 3. Send the chunks to the embedding model
# 4. Save the embeddings to a vector database
# 5. Perform similarity search on the vector database to find similar documents
# 6. retrieve the similar documents and present them to the user
## run pip install -r requirements.txt to install the required packages

# Importing the tool which will read the document.
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

doc_path = "/media/aumoza/Strg_1/ollama-finetune/ollamaFreeCodeCamp/BOI.pdf"
model = "llama3.2:1B "

if doc_path:
    loader = UnstructuredPDFLoader(file_path=doc_path)
    data = loader.load()
    print("Done Loading.")
else:
    print("Upload a PDF File")

content = data[0].page_content # This is the preview for the first page.
print(content[:100])

# So now we're extracting the text from the pdf file and splitting into small chunks.
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Now we're starting with splitting and chunking :
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1200, chunk_overlap = 300)
chunks = text_splitter.split_documents(data)
print("Done splitting...")

# print(f"Number of chunks : {len(chunks)}")
# print(f"Example of chunk : {chunks[0]}")

# Adding to vector database:
import ollama
ollama.pull('nomic-embed-text') # Pulling an embed model.

# Creating a vector database :
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model='nomic-embed-text'),
    collection_name='simple_rag'
)

print("Done adding to vector databse")

# Doing the retrieval
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Setting up the model to use :
llm = ChatOllama(model=model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
                You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}
"""
)