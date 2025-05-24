from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from chatbot_backend.tools.insurance_pdf import InsurancePDFTool
from chatbot_backend.tools.angelone_support import AngelOneLLMRetriever
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr
import requests

# Load insurance PDF tool (vectorstore-based)
insurance_tool = InsurancePDFTool()


# CombinedRetriever that uses both insurance PDF vectorstore and AngelOne LLM retriever
class CombinedRetriever(BaseRetriever):
    _retrievers = PrivateAttr()

    def __init__(self, retrievers):
        super().__init__()
        self._retrievers = retrievers

    def get_relevant_documents(self, query):
        docs = []
        for retriever in self._retrievers:
            docs.extend(retriever.get_relevant_documents(query))
        return docs

    async def aget_relevant_documents(self, query):
        return self.get_relevant_documents(query)

llm = ChatGroq(
    model="llama3-8b-8192"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful customer support agent for AngelOne and insurance queries. "
        "If the user greets you (e.g., says 'hi', 'hello', 'hey', 'what can you do', etc.), respond with a friendly greeting and a short description of your capabilities (you need not to retrieve any of the context in this case): "
        "'Hello! I am a customer support assistant. You can ask me questions about insurance policies, claim processes, AngelOne account support, refunds, account verification, and more. I will answer using the latest information from AngelOne support and insurance documents.' "
        "Otherwise, you must derive the answer using only the provided context. "
        "If the answer is not present in the context or could not be derived from it, reply with 'I Don't know'. "
    )),
    ("user", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
])

# Use both retrievers: insurance PDF vectorstore and AngelOne LLM-based retriever
angelone_llm_retriever = AngelOneLLMRetriever()
combined_retriever = CombinedRetriever([insurance_tool, angelone_llm_retriever])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=combined_retriever,
    chain_type="stuff",
    return_source_documents=False,
    chain_type_kwargs={
        "prompt": prompt
    }
)

def answer_question(query):
    return qa_chain.invoke(query)
