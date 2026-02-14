"""
Vector store: MongoDB Atlas with Vector Search. Insert embeddings, search by similarity.
Requires a vector index on the collection (see docs/ATLAS_VECTOR_INDEX.md).
"""


import os
from typing import Any, List, Union


try:
   from dotenv import load_dotenv
   load_dotenv()
except ImportError:
   pass


import numpy as np
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection


try:
   import certifi
except ImportError:
   certifi = None




def get_client(uri: Union[str, None] = None, server_selection_timeout_ms: Union[int, None] = None) -> MongoClient:
   """Create and return a MongoDB client. Optional server_selection_timeout_ms to avoid long hangs (e.g. 5000)."""
   uri = (uri or os.environ.get("MONGODB_URI", "mongodb://localhost:27017")).strip().strip('"\'')
   if not (uri.startswith("mongodb://") or uri.startswith("mongodb+srv://")):
       raise ValueError(
           "MONGODB_URI must start with 'mongodb://' or 'mongodb+srv://'. "
           "Check your .env — e.g. MONGODB_URI=mongodb+srv://USERNAME:PASSWORD@host.mongodb.net/"
       )
   kwargs = {}
   if certifi is not None and "mongodb+srv://" in uri:
       kwargs["tlsCAFile"] = certifi.where()
   if server_selection_timeout_ms is not None:
       kwargs["serverSelectionTimeoutMS"] = server_selection_timeout_ms
   return MongoClient(uri, **kwargs)




def get_db(client: Union[MongoClient, None] = None, db_name: Union[str, None] = None) -> Database:
   """Get database. Uses MONGODB_DB env var or default 'treehacks' if db_name not given."""
   if client is None:
       client = get_client()
   name = db_name or os.environ.get("MONGODB_DB", "treehacks")
   return client[name]




def get_collection(
   collection_name: str = "frames",
   db: Union[Database, None] = None,
   client: Union[MongoClient, None] = None,
) -> Collection:
   """Get the frames collection."""
   if db is None:
       db = get_db(client=client)
   return db[collection_name]




def insert_frame(
   embedding: Union[List[float], np.ndarray],
   collection: Union[Collection, None] = None,
   collection_name: str = "frames",
   **extra: Any,
) -> Any:
   """
   Insert a frame: store an embedding vector and optional metadata.


   Args:
       embedding: Vector (e.g. from an encoder). Dimension must match your Atlas vector index.
       collection: Collection to use. If None, uses default.
       collection_name: Collection name when collection is None.
       **extra: Extra fields (e.g. frame_id, timestamp).


   Returns:
       Inserted document's _id.
   """
   if collection is None:
       collection = get_collection(collection_name=collection_name)


   doc = {"embedding": list(embedding), **extra}
   result = collection.insert_one(doc)
   return result.inserted_id




def search(
   query_embedding: Union[List[float], np.ndarray],
   collection: Union[Collection, None] = None,
   collection_name: str = "frames",
   index_name: str = "frame_vectors",
   top_k: int = 10,
   num_candidates: Union[int, None] = None,
   filter_query: Union[dict, None] = None,
   projection: Union[dict, None] = None,
) -> List[dict]:
   """
   Vector search: find nearest neighbors to the query embedding (Atlas Vector Search).


   Requires a vector index on the collection. See docs/ATLAS_VECTOR_INDEX.md.


   Args:
       query_embedding: Query vector (same dimension as index).
       collection: Collection. If None, uses default.
       collection_name: Collection name when collection is None.
       index_name: Atlas Vector Search index name (default "frame_vectors").
       top_k: Max results to return.
       num_candidates: ANN candidates (default 20 * top_k).
       filter_query: Optional MQL filter.
       projection: Optional field projection.


   Returns:
       List of docs with "score" (similarity), sorted by score descending.
   """
   if collection is None:
       collection = get_collection(collection_name=collection_name)


   query_vec = list(np.asarray(query_embedding, dtype=np.float64))
   limit = int(top_k)
   num_candidates = num_candidates if num_candidates is not None else max(limit * 20, 100)


   stage: dict = {
       "index": index_name,
       "path": "embedding",
       "queryVector": query_vec,
       "limit": limit,
       "numCandidates": num_candidates,
   }
   if filter_query:
       stage["filter"] = filter_query


   pipeline: List[dict] = [{"$vectorSearch": stage}]
   if projection:
       pipeline.append({"$project": projection})
   pipeline.append({"$addFields": {"score": {"$meta": "vectorSearchScore"}}})


   cursor = collection.aggregate(pipeline)
   return list(cursor)




def create_vector_index(
   num_dimensions: int = 5,
   index_name: str = "frame_vectors",
   collection_name: str = "frames",
   db: Union[Database, None] = None,
   client: Union[MongoClient, None] = None,
) -> dict:
   """
   Create the Atlas Vector Search index on the collection (if it doesn't exist).
   Uses createSearchIndexes. Index may take a minute to become ready.


   Args:
       num_dimensions: Embedding size (default 5 for test_db.py; use 384, 768, etc. for real encoders).
       index_name: Index name (default "frame_vectors").
       collection_name: Collection to index (default "frames").
       db: Database. If None, uses default from env.
       client: Client. If None, uses default.


   Returns:
       Result of createSearchIndexes (e.g. {"ok": 1, "indexesCreated": [...]}).
   """
   if db is None:
       db = get_db(client=client)
   cmd = {
       "createSearchIndexes": collection_name,
       "indexes": [
           {
               "name": index_name,
               "type": "vectorSearch",
               "definition": {
                   "fields": [
                       {
                           "type": "vector",
                           "path": "embedding",
                           "numDimensions": num_dimensions,
                           "similarity": "cosine",
                       }
                   ]
               },
           }
       ],
   }
   return db.command(cmd)