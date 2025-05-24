from langchain_groq import ChatGroq
from langchain_core.retrievers import BaseRetriever
import requests
from bs4 import BeautifulSoup
from pydantic import PrivateAttr

SUPPORT_URL = "https://www.angelone.in/support"

class AngelOneLLMRetriever(BaseRetriever):
    _llm = PrivateAttr()
    _support_url = PrivateAttr()
    _max_pages = PrivateAttr()

    def __init__(self, support_url=SUPPORT_URL, max_pages=20):
        super().__init__()
        self._llm = ChatGroq(model="llama3-8b-8192")
        self._support_url = support_url
        self._max_pages = max_pages  # To avoid excessive crawling

    def _get_all_support_links(self):
        resp = requests.get(self._support_url)
        soup = BeautifulSoup(resp.text, "html.parser")
        links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/support"):
                links.add("https://www.angelone.in" + href)
            elif href.startswith(self._support_url):
                links.add(href)
        # Limit the number of pages to avoid excessive requests
        return list(links)[:self._max_pages] or [self._support_url]

    def get_relevant_documents(self, query):
        support_links = self._get_all_support_links()
        relevant_chunks = []
        for url in support_links:
            try:
                resp = requests.get(url, timeout=10)
                soup = BeautifulSoup(resp.text, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                # Ask LLM if this page contains relevant info
                check_prompt = (
                    "You are a helpful customer support agent. "
                    "Given the following support page text, does it contain information that could help answer the user's question? "
                    "If yes, extract the most relevant paragraph(s) or sentences. If not, reply with 'NO RELEVANT INFO'.\n"
                    "User question: {question}\n\nSupport Page Text:\n{text}\n\nRelevant Information:"
                )
                chunk = self._llm.invoke(check_prompt.format(question=query, text=text))
                if chunk.strip().lower() != "no relevant info":
                    relevant_chunks.append(chunk.strip())
            except Exception:
                continue
        # Now, ask LLM to synthesize the final answer from all relevant chunks
        if not relevant_chunks:
            final_answer = "I Don't know."
        else:
            synth_prompt = (
                "You are a helpful customer support agent. "
                "Given the following extracted information from multiple support pages, answer the user's question as best as possible. "
                "If the answer is not present, reply with 'I Don't know'.\n"
                "User question: {question}\n\nExtracted Information:\n{chunks}\n\nAnswer:"
            )
            final_answer = self._llm.invoke(synth_prompt.format(question=query, chunks="\n\n".join(relevant_chunks)))
        from langchain_core.documents import Document
        return [Document(page_content=final_answer, metadata={"source": self._support_url})]

    async def aget_relevant_documents(self, query):
        return self.get_relevant_documents(query)


# class AngelOneSupportTool:
#     def __init__(self, persist_path="faiss_angelone_support"):
#         self.persist_path = persist_path
#         self.embeddings = OllamaEmbeddings(
#             model="llama3"
#         )
#         self.vectorstore = None
#         self._load_or_create_vectorstore()

#     def _get_all_support_links(self):
#         resp = requests.get(SUPPORT_URL)
#         soup = BeautifulSoup(resp.text, "html.parser")
#         links = set()
#         for a in soup.find_all("a", href=True):
#             href = a["href"]
#             if href.startswith("/support"):
#                 links.add("https://www.angelone.in" + href)
#             elif href.startswith(SUPPORT_URL):
#                 links.add(href)
#         return list(links) or [SUPPORT_URL]

#     def _extract_faq_text(self, html):
#         """
#         Extract FAQ/helpful text from the HTML using BeautifulSoup.
#         Looks for elements with FAQ, question, answer, or help in class/id, or main content blocks.
#         Always includes all <p> tags with enough text.
#         """
#         soup = BeautifulSoup(html, "html.parser")
#         faqs = set()
#         # Look for common FAQ/helpful content containers
#         for tag in soup.find_all(["section", "div", "article", "li"]):
#             cls = tag.get("class") or []
#             id_ = tag.get("id") or ""
#             cls_str = " ".join(cls).lower()
#             id_str = str(id_).lower()
#             text = tag.get_text(strip=True)
#             if not text or len(text) < 30:
#                 continue
#             # Heuristics: FAQ, question, answer, help, support, or main content
#             if ("faq" in cls_str or "faq" in id_str or
#                 "question" in cls_str or "question" in id_str or
#                 "answer" in cls_str or "answer" in id_str or
#                 "help" in cls_str or "help" in id_str or
#                 "support" in cls_str or "support" in id_str or
#                 "content" in cls_str or "content" in id_str or
#                 "main" in cls_str or "main" in id_str):
#                 faqs.add(text)
#         # Always include all <p> tags with enough text
#         for p in soup.find_all("p"):
#             text = p.get_text(strip=True)
#             if text and len(text) > 40:
#                 faqs.add(text)
#         return list(faqs)

#     def _load_or_create_vectorstore(self):
#         if os.path.exists(self.persist_path):
#             self.vectorstore = FAISS.load_local(self.persist_path, self.embeddings, allow_dangerous_deserialization=True)
#         else:
#             docs = []
#             links = self._get_all_support_links()
#             print(f"Found {len(links)} support links to store on https://www.angleone.in/support")
#             for url in links:
#                 try:
#                     resp = requests.get(url)
#                     faqs = self._extract_faq_text(resp.text)
#                     for faq in faqs:
#                         # Use LangChain's Document format
#                         from langchain_core.documents import Document
#                         docs.append(Document(page_content=faq, metadata={"source": url}))
#                 except Exception:
#                     print(f"Failed to load {url}. Skipping...")
#                     continue
#             # Save all document texts to a .txt file for inspection
#             txt_path = os.path.join(os.path.dirname(self.persist_path), "angelone_support_embedded_docs.txt")
#             with open(txt_path, "w", encoding="utf-8") as f:
#                 for doc in docs:
#                     f.write(doc.page_content + "\n" + ("-"*80) + "\n")
#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#             splits = text_splitter.split_documents(docs)
#             self.vectorstore = FAISS.from_documents(splits, self.embeddings)
#             self.vectorstore.save_local(self.persist_path)

#     def search(self, query, k=4):
#         return self.vectorstore.similarity_search(query, k=k)
