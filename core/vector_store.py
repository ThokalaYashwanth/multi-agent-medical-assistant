import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from core.config import settings
from core.llm import get_embeddings


class VectorStore:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.index_path = settings.FAISS_INDEX_PATH
        self.db = None

    def load_or_build(self):
        """Load existing FAISS index or build from documents."""
        if os.path.exists(self.index_path):
            self.db = FAISS.load_local(
                self.index_path, self.embeddings, allow_dangerous_deserialization=True
            )
            print(f"[VectorStore] Loaded index from {self.index_path}")
        else:
            self._build_index()
        return self.db

    def _build_index(self):
        """Ingest medical documents and build FAISS index."""
        docs_path = settings.MEDICAL_DOCS_PATH
        if not os.path.exists(docs_path):
            os.makedirs(docs_path, exist_ok=True)
            print(f"[VectorStore] Created empty docs folder at {docs_path}. Add PDF/TXT files and rebuild.")
            self.db = FAISS.from_texts(["placeholder"], self.embeddings)
            return

        loader = PyPDFDirectoryLoader(docs_path)
        raw_docs: List[Document] = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "],
        )
        chunks = splitter.split_documents(raw_docs)
        print(f"[VectorStore] Indexed {len(chunks)} chunks from {len(raw_docs)} documents")

        self.db = FAISS.from_documents(chunks, self.embeddings)
        self.db.save_local(self.index_path)

    def retriever(self, k: int = None):
        k = k or settings.TOP_K_RETRIEVAL
        return self.db.as_retriever(search_kwargs={"k": k})


# Singleton
_store = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
        _store.load_or_build()
    return _store
