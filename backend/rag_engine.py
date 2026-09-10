import os
import requests
from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

class ProjectRAGEngine:
    def __init__(self, project_id: str, storage_base="./storage/projects"):
        self.project_id = project_id
        self.project_dir = os.path.join(storage_base, project_id)
        self.db_dir = os.path.join(self.project_dir, "chroma_db")
        self.uploads_dir = os.path.join(self.project_dir, "uploads")
        
        os.makedirs(self.db_dir, exist_ok=True)
        os.makedirs(self.uploads_dir, exist_ok=True)
        
        my_api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash", 
            temperature=0.1,
            google_api_key=my_api_key,
            streaming=True
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2", 
            google_api_key=my_api_key
        )

    def get_vectorstore(self):
        return Chroma(
            persist_directory=self.db_dir,
            embedding_function=self.embeddings
        )

    def _expand_query(self, context_text: str) -> str:
        prompt_template = """
        You are an expert academic research assistant. Given the following project goal/context, 
        convert it into a strict Boolean search string for OpenAlex using relevant academic and scientific terminology. 
        Use ONLY AND, OR, and quotes. Output ONLY the boolean string.
        
        Project Context: {context}
        """
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context_text}).strip().strip('`').strip()

    def _reconstruct_abstract(self, inverted_index) -> str:
        if not inverted_index:
            return "No abstract available."
        max_index = max([max(pos) for pos in inverted_index.values()])
        words = [""] * (max_index + 1)
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word
        return " ".join(words)

    def auto_fetch_papers(self, project_context: str, max_results: int = 20, year_from: int = None):
        boolean_query = self._expand_query(project_context)
        url = "https://api.openalex.org/works"
        params = {
            "search": boolean_query,
            "per_page": max_results,
            "select": "title,publication_year,abstract_inverted_index"
        }
        if year_from:
            params["filter"] = f"publication_year:>{year_from}"
        
        res = requests.get(url, params=params)
        docs = []
        if res.status_code == 200:
            for paper in res.json().get('results', []):
                title = paper.get('title', 'Unknown Title')
                year = paper.get('publication_year', 'Unknown Year')
                abstract = self._reconstruct_abstract(paper.get('abstract_inverted_index'))
                
                doc = Document(
                    page_content=f"Title: {title}\nYear: {year}\nAbstract: {abstract}",
                    metadata={"source": "OpenAlex", "title": title, "year": year}
                )
                docs.append(doc)
                
        if docs:
            vectorstore = self.get_vectorstore()
            vectorstore.add_documents(docs)
        return len(docs), boolean_query

    def add_pdf(self, file_bytes: bytes, filename: str):
        pdf_path = os.path.join(self.uploads_dir, filename)
        with open(pdf_path, "wb") as f:
            f.write(file_bytes)
            
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_text(full_text)
        
        docs = [
            Document(
                page_content=chunk,
                metadata={"source": "Uploaded PDF", "filename": filename}
            )
            for chunk in chunks
        ]
        
        if docs:
            vectorstore = self.get_vectorstore()
            vectorstore.add_documents(docs)
        return len(docs)

    def list_documents(self):
        collection = self.get_vectorstore()._collection
        data = collection.get(include=["metadatas"])
        
        unique_docs = {}
        for meta in data.get("metadatas", []):
            if not meta: continue
            
            if meta.get("source") == "OpenAlex":
                name = meta.get("title")
                doc_type = "OpenAlex API"
            else:
                name = meta.get("filename")
                doc_type = "Uploaded PDF"
                
            if name and name not in unique_docs:
                unique_docs[name] = {"name": name, "type": doc_type}
                
        return list(unique_docs.values())

    def delete_document(self, doc_name: str):
        collection = self.get_vectorstore()._collection
        try:
            collection.delete(where={"title": doc_name})
        except:
            pass
        try:
            collection.delete(where={"filename": doc_name})
        except:
            pass

    def chat_stream(self, user_question: str):
        """Streams answers prioritizing local RAG context, falling back gracefully to general knowledge."""
        vectorstore = self.get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        
        docs = retriever.invoke(user_question)
        context_str = "\n\n".join([d.page_content for d in docs])
        
        qa_prompt = ChatPromptTemplate.from_template("""
        You are an expert research assistant.
        
        First, answer using the provided context from research papers and uploaded documents.
        Always cite paper titles, DOIs, or PDF filenames when making claims supported by context.

        If the context does not contain the complete answer, or the user asks general questions 
        (e.g., basic concepts, broad overviews), supplement with your general knowledge. 
        Note when information comes from general knowledge rather than project sources.

        Context: {context}
        Question: {input}
        """)

        chain = qa_prompt | self.llm | StrOutputParser()
        for chunk in chain.stream({"context": context_str, "input": user_question}):
            yield chunk

    # ==========================================
    # MODEL 1: RESEARCH (Explicit OpenAlex Search)
    # ==========================================
    def search_and_add_openalex(self, query: str, max_results: int = 20, year_from: int = None):
        url = "https://api.openalex.org/works"
        params = {
            "search": query,
            "per_page": max_results,
            "select": "title,publication_year,abstract_inverted_index,doi"
        }
        if year_from:
            params["filter"] = f"publication_year:>{year_from}"
        
        res = requests.get(url, params=params)
        docs = []
        if res.status_code == 200:
            for paper in res.json().get('results', []):
                title = paper.get('title', 'Unknown Title')
                year = paper.get('publication_year', 'Unknown Year')
                doi = paper.get('doi', '')
                abstract = self._reconstruct_abstract(paper.get('abstract_inverted_index'))
                
                doc = Document(
                    page_content=f"Title: {title}\nYear: {year}\nDOI: {doi}\nAbstract: {abstract}",
                    metadata={"source": "OpenAlex", "title": title, "year": year, "doi": doi}
                )
                docs.append(doc)
                
        if docs:
            vectorstore = self.get_vectorstore()
            vectorstore.add_documents(docs)
            
        return len(docs)

    # ==========================================
    # MODEL 2: ASK (Context + Knowledge Stream)
    # ==========================================
    def chat_stream(self, user_question: str):
        """Streams answers prioritizing local RAG context, falling back gracefully to general knowledge."""
        vectorstore = self.get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        
        docs = retriever.invoke(user_question)
        context_str = "\n\n".join([d.page_content for d in docs])
        
        qa_prompt = ChatPromptTemplate.from_template("""
        You are an expert paleontology research assistant.
        
        First, answer using the provided context from research papers and uploaded documents.
        Always cite paper titles, DOIs, or PDF filenames when making claims supported by context.

        If the context does not contain the complete answer, or the user asks general questions 
        (e.g., basic concepts, broad overviews), supplement with your general paleontology knowledge. 
        Note when information comes from general knowledge rather than project sources.

        Context: {context}
        Question: {input}
        """)

        chain = qa_prompt | self.llm | StrOutputParser()
        for chunk in chain.stream({"context": context_str, "input": user_question}):
            yield chunk

    # ==========================================
    # MODEL 3: EVIDENCE MAP (Consensus & Conflict Matrix)
    # ==========================================
    def generate_evidence_map(self, topic: str):
        """Analyzes literature chunks to map supporting vs. conflicting evidence for a specific topic."""
        vectorstore = self.get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 12})
        
        docs = retriever.invoke(f"claims evidence analysis regarding {topic}")
        context_str = "\n\n".join([d.page_content for d in docs])

        map_prompt = ChatPromptTemplate.from_template("""
        Analyze the research context and build an Evidence Map for: "{topic}"
        
        Group hypotheses/claims into distinct claims. For each claim, identify:
        1. Claim statement.
        2. Supporting evidence (with cited sources).
        3. Conflicting or contrasting evidence (with cited sources).
        4. Current consensus status ("Strong Consensus", "Contested", or "Insufficient Data").

        Return strictly valid JSON with this format:
        {{
          "topic": "{topic}",
          "claims": [
            {{
              "claim": "Claim title or hypothesis",
              "supporting_evidence": ["Evidence 1 (Source)", "Evidence 2 (Source)"],
              "conflicting_evidence": ["Counter-argument 1 (Source)"],
              "consensus": "Contested"
            }}
          ]
        }}

        Context: {context}
        """)

        chain = map_prompt | self.llm | JsonOutputParser()
        return chain.invoke({"topic": topic, "context": context_str})