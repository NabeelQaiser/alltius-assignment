from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import glob

PDF_DIR = os.path.join(os.path.dirname(__file__), '../../resources/insurance-pdfs')

class InsurancePDFTool:
    def __init__(self, persist_path="faiss_insurance_pdf"):
        self.persist_path = persist_path
        self.embeddings = OllamaEmbeddings(
            model="llama3"
        )
        self.vectorstore = None
        self._load_or_create_vectorstore()

    def _load_or_create_vectorstore(self):
        if os.path.exists(self.persist_path):
            self.vectorstore = FAISS.load_local(self.persist_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            docs = []
            for pdf_file in glob.glob(os.path.join(PDF_DIR, "*.pdf")):
                loader = PyPDFLoader(pdf_file)
                docs.extend(loader.load())
            # Save all document texts to a .txt file for inspection
            txt_path = os.path.join(os.path.dirname(self.persist_path), "insurance_pdf_embedded_docs.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                for doc in docs:
                    f.write(doc.page_content + "\n" + ("-"*80) + "\n")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
            self.vectorstore.save_local(self.persist_path)

    def search(self, query, k=4):
        return self.vectorstore.similarity_search(query, k=k)

    def get_relevant_documents(self, query):
        # For compatibility with LangChain retriever interface
        return self.search(query)
