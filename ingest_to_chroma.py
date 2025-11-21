import json
import chromadb
from chromadb.utils import embedding_functions

# --- Configuration ---
JSON_FILE_PATH = 'products/products_metadata_with_embeddings.json'
CHROMA_DB_PATH = './chroma_db' # Directory where ChromaDB will store data
COLLECTION_NAME = 'products'
# The embedding size (dimension) is inferred from the first vector in the file.
# Assuming the file is named products_metadata_with_embeddings.json

def load_and_ingest_data():
    """
    Loads product data from a JSON file and ingests it into ChromaDB.
    """
    print(f"1. Loading data from {JSON_FILE_PATH}...")
    try:
        with open(JSON_FILE_PATH, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found at {JSON_FILE_PATH}. Please ensure the file is in the correct directory.")
        return
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {JSON_FILE_PATH}. Please check the file format.")
        return

    if not data or not isinstance(data, list):
        print("ERROR: JSON file is empty or not a list of product objects.")
        return
        
    print(f"Successfully loaded {len(data)} product records.")

    # --- Data Preparation ---
    ids = []
    embeddings = []
    metadatas = []
    
    # Infer the dimension (size) of the vector from the first product
    vector_dimension = len(data[0].get('vector', []))
    if vector_dimension == 0:
        print("ERROR: The first product does not contain a 'vector' field or the vector is empty.")
        return

    print(f"2. Preparing data for ingestion (Inferred vector dimension: {vector_dimension})...")

    for item in data:
        product_id = item.get('id')
        vector = item.get('vector')
        
        # Ensure 'id' and 'vector' exist and are valid
        if not product_id or not vector or len(vector) != vector_dimension:
            print(f"Skipping record due to missing/invalid data: {item}")
            continue

        ids.append(str(product_id))
        embeddings.append(vector)
        
        # Create metadata dictionary by excluding 'id' and 'vector'
        metadata = {k: v for k, v in item.items() if k not in ['id', 'vector']}
        # Convert any list/dict metadata fields to string if they are too complex for direct storage
        metadata_safe = {}
        for k, v in metadata.items():
             # Basic safety check for value types that might not be supported by Chroma metadata
            if isinstance(v, (list, dict)):
                metadata_safe[k] = json.dumps(v)
            else:
                metadata_safe[k] = str(v)
                
        metadatas.append(metadata_safe)

    if not ids:
        print("ERROR: No valid product records were found after filtering.")
        return
        
    # --- ChromaDB Initialization and Ingestion ---
    print(f"3. Initializing ChromaDB client at {CHROMA_DB_PATH}...")
    
    # Initialize the client with the persistent storage path
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Since the embeddings are pre-calculated, we use a NullEmbeddingFunction.
    # We must also specify the dimension for the collection.
    # Note: ChromaDB handles dimension automatically on the first add, 
    # but defining a NullEmbeddingFunction signals that no model needs to be run.
    class NullEmbeddingFunction(embedding_functions.EmbeddingFunction):
        def __call__(self, texts, images=None):
            raise NotImplementedError("This is a Null function; embeddings are pre-calculated.")

    # Create or get the collection. 
    # Chroma requires a distance function. 'cosine' is standard for embeddings.
    print(f"4. Getting or creating collection '{COLLECTION_NAME}'...")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=NullEmbeddingFunction(), # Use the Null function
        metadata={"hnsw:space": "cosine"} # Specify the distance metric
    )

    # Add the data to the collection
    print(f"5. Ingesting {len(ids)} records into ChromaDB...")
    try:
        collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print("✅ Ingestion complete!")
        print(f"Total documents in collection '{COLLECTION_NAME}': {collection.count()}")
    except Exception as e:
        print(f"ERROR during ingestion: {e}")

if __name__ == "__main__":
    load_and_ingest_data()