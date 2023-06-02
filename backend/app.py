from flask import Flask
from flask import request, jsonify,render_template,jsonify
import os
import langchain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import VectorDBQA
from langchain.vectorstores import Chroma
from flask_cors import CORS
server = Flask(__name__)
CORS(server)

def Get_Answer(question):
    os.environ["OPENAI_API_KEY"] = "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"
    loader = PyPDFLoader("./content/constitution.pdf")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    qa = VectorDBQA.from_chain_type(llm=ChatOpenAI(), chain_type="stuff", vectorstore=db, k=1)
    ans=qa.run(question)
    return ans


@server.route('/', methods=['POST'])
def hello_world():
    quest = request.json['first_name']
    ans = Get_Answer(quest)
    data={"answer":ans}
    return jsonify(data)
if __name__ == '__main__':
   server.run(host="0.0.0.0",debug = True)