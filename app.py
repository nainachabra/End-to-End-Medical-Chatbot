from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain_groq.chat_models import ChatGroq
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY=os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_hugging_face_embeddings()


index_name = "medicalbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


llm = ChatGroq(
    api_key="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",
    model="mixtral-8x7b-32768",
    temperature=0.7,
    max_tokens=500
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

ustom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Context: {context}
    Question: {question}
    Answer concisely and accurately.
    """
)

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,  # Pass the language model
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt},  # Provide the custom prompt
)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = qa_chain.invoke({"query": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
