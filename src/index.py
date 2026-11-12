"""Build or load the LlamaIndex vector index over data/experience.
 
Run directly to (re)build the index and test retrieval:
    python src/index.py
Re-run it any time you edit files in data/experience.
"""
from pathlib import Path

from llama_index.core import(
    Settings, 
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "experience"
STORAGE_DIR = ROOT / "storage"

# Free local embedding model (downloads ~130 MB on first run).
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# LlamaIndex only does retrieval here; Claude is called through LangChain later.
Settings.llm = None


def build_index(rebuild: bool = False) -> VectorStoreIndex:
    """Load the saved index, or build it from data/experience."""
    if STORAGE_DIR.exists() and not rebuild:
        storage = StorageContext.from_defaults(persist_dir = str(STORAGE_DIR))
        return load_index_from_storage(storage)

    docs = SimpleDirectoryReader(str(DATA_DIR), required_exts = 
                                 [".md"]).load_data()
   # Split on markdown headings so each section (e.g. "LLM inference optimization")
   # becomes its own retrievable chunk.
    
    index = VectorStoreIndex.from_documents(docs, transformations = [MarkdownNodeParser()])
    print(f"Indexed {len(docs)} files -> {STORAGE_DIR}")
    return index


def get_retriever(top_k: int = 5):
    return build_index().as_retriever(similarity_tok_k=top_k)

if __name__ == "__main__":
    build_index(rebuild=True)
    retriever = get_retriever(top_k=3)

    query = "LLM inference optimization and latency reduction"
    print(f"\nQuery: {query}\n")
    for hit in retriever.retrieve(query):
        source = hit.node.metadata.get("file_name", "?")
        snippet = hit.node.get_content()[:150].replace("\n", " ")
        print(f"[{hit.score:.2f}] {source}: {snippet}...")
