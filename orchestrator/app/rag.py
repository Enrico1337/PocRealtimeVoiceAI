"""
RAG module: Qdrant vector store + BGE-M3 embeddings + knowledge base ingestion.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer

from .settings import Settings

logger = logging.getLogger(__name__)


class RAGService:
    """RAG service for knowledge base retrieval."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client: Optional[QdrantClient] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.collection_name = settings.rag_collection_name
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Qdrant client and embedding model."""
        if self._initialized:
            return

        logger.info("Initializing RAG service...")

        # Initialize Qdrant client
        self.client = QdrantClient(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
        )

        # Load embedding model (CPU is fine for POC)
        logger.info(f"Loading embedding model: {self.settings.embedding_model}")
        self.embedding_model = SentenceTransformer(
            self.settings.embedding_model,
            device="cpu"
        )

        # Get embedding dimension
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

        # Ensure collection exists
        await self._ensure_collection()

        # Ingest knowledge base
        await self._ingest_knowledge_base()

        self._initialized = True
        logger.info("RAG service initialized successfully")

    async def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self.embedding_dim,
                    distance=qdrant_models.Distance.COSINE
                )
            )
        else:
            logger.info(f"Collection {self.collection_name} already exists")

    async def _ingest_knowledge_base(self) -> None:
        """Ingest documents from the knowledge base directory."""
        kb_path = Path(self.settings.kb_path)

        if not kb_path.exists():
            logger.warning(f"Knowledge base path does not exist: {kb_path}")
            return

        # Get all text files
        files = list(kb_path.glob("**/*.md")) + \
                list(kb_path.glob("**/*.txt")) + \
                list(kb_path.glob("**/*.rst"))

        if not files:
            logger.warning("No documents found in knowledge base")
            return

        logger.info(f"Found {len(files)} documents in knowledge base")

        # Check if collection already has data
        collection_info = self.client.get_collection(self.collection_name)
        if collection_info.points_count > 0:
            logger.info(
                f"Collection already has {collection_info.points_count} points, "
                "skipping ingestion"
            )
            return

        # Process each file
        all_chunks = []
        all_metadata = []

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
                chunks = self._chunk_text(content, file_path.name)

                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadata.append({
                        "source": file_path.name,
                        "chunk_index": i,
                    })

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        if not all_chunks:
            logger.warning("No chunks generated from knowledge base")
            return

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = self.embedding_model.encode(
            all_chunks,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Upsert to Qdrant
        logger.info("Upserting to Qdrant...")
        points = [
            qdrant_models.PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload={
                    "text": all_chunks[i],
                    **all_metadata[i]
                }
            )
            for i in range(len(all_chunks))
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        logger.info(f"Ingested {len(points)} chunks into Qdrant")

    def _chunk_text(self, text: str, source: str) -> list[str]:
        """Split text into overlapping chunks."""
        chunk_size = self.settings.chunk_size
        chunk_overlap = self.settings.chunk_overlap

        # Clean text
        text = text.strip()
        if not text:
            return []

        # Split into sentences (simple approach)
        sentences = []
        for paragraph in text.split("\n\n"):
            paragraph = paragraph.strip()
            if paragraph:
                # Split on sentence boundaries
                for sent in paragraph.replace(". ", ".|").split("|"):
                    sent = sent.strip()
                    if sent:
                        sentences.append(sent)

        if not sentences:
            return []

        # Build chunks from sentences
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))

                # Start new chunk with overlap
                overlap_text = " ".join(current_chunk)
                if len(overlap_text) > chunk_overlap:
                    # Keep last part as overlap
                    words = overlap_text.split()
                    overlap_words = []
                    overlap_len = 0
                    for word in reversed(words):
                        if overlap_len + len(word) > chunk_overlap:
                            break
                        overlap_words.insert(0, word)
                        overlap_len += len(word) + 1
                    current_chunk = overlap_words
                    current_length = overlap_len
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_length + 1

        # Don't forget last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    async def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Retrieve relevant documents for a query.

        Returns list of dicts with 'text', 'source', 'score' keys.
        """
        if not self._initialized:
            logger.warning("RAG service not initialized")
            return []

        top_k = top_k or self.settings.rag_top_k

        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True
        ).tolist()

        # Search Qdrant
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )

        # Format results
        documents = []
        for result in results:
            documents.append({
                "text": result.payload.get("text", ""),
                "source": result.payload.get("source", "unknown"),
                "score": result.score
            })

        logger.debug(f"Retrieved {len(documents)} documents for query")
        return documents

    def format_context(self, documents: list[dict]) -> str:
        """Format retrieved documents as context for the LLM."""
        if not documents:
            return ""

        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(
                f"[Quelle {i}: {doc['source']}]\n{doc['text']}"
            )

        return "\n\n".join(context_parts)

    async def close(self) -> None:
        """Close connections."""
        if self.client:
            self.client.close()
            self.client = None
        self._initialized = False
        logger.info("RAG service closed")
