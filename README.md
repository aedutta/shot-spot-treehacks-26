# treehacks-2026

**Vector store** on MongoDB Atlas: store embeddings (e.g. from video frames) and search by similarity using Atlas Vector Search.

## Features

- **insert_frame** — Store an embedding vector and optional metadata.
- **search** — Vector search (nearest neighbors) via Atlas Vector Search index.

Requires a **vector index** on your collection. See **[docs/ATLAS_VECTOR_INDEX.md](docs/ATLAS_VECTOR_INDEX.md)**.