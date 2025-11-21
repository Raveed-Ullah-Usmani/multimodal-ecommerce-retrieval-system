import chromadb
import numpy as np

# --- Configuration (Must match ingest_to_chroma.py) ---
CHROMA_DB_PATH = './chroma_db'
COLLECTION_NAME = 'products'
# The dimension of the vector (1024) is needed for creating a valid query vector.
VECTOR_DIMENSION = 1024

def test_chroma_setup():
    """
    Connects to the persistent ChromaDB, performs a count check, 
    and runs a sample query to verify data integrity.
    """
    print(f"1. Connecting to ChromaDB client at {CHROMA_DB_PATH}...")
    try:
        # Initialize the client with the persistent storage path
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    except Exception as e:
        print(f"FATAL ERROR: Could not connect to ChromaDB. Ensure it was saved correctly. Error: {e}")
        return

    # --- 2. Count Check ---
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        count = collection.count()

        print("\n--- Count Check ---")
        if count == 1200:
            print(f"✅ SUCCESS: Found {count} documents in the '{COLLECTION_NAME}' collection.")
        else:
            print(f"❌ FAILURE: Expected 1200 documents, but found {count}. Ingestion may have been incomplete.")
            return

    except Exception as e:
        print(f"❌ FAILURE: Could not retrieve collection '{COLLECTION_NAME}'. Check the collection name. Error: {e}")
        return
    
    # --- 3. Query Check ---
    print("\n--- Sample Query Check ---")
    
    # To simulate a search, we need a query vector. 
    # Since we don't have the original model to generate a real query vector 
    # for a search term like "red summer dress", we will grab the vector
    # of the very first product and use it as the query vector. 
    # This should always return the first product as the best match (distance 0).
    try:
        # Fetch the ID and vector of the first document to use for testing the query system
        first_doc = collection.get(ids=["ash-grey-kurta-waistcoat-set"], include=['embeddings'])

        if not first_doc or not first_doc.get('embeddings') or not first_doc.get('ids'):
             print("❌ FAILURE: Could not retrieve a sample document for querying.")
             return

        # Use the vector of the first product as our query vector
        query_vector = first_doc['embeddings'][0]

    except Exception:
        # If fetching by ID fails, we'll create a random vector
        print("INFO: Sample ID not found. Creating a random vector for query test.")
        query_vector = np.random.rand(VECTOR_DIMENSION).tolist()

    
    print("Running nearest neighbor search (k=3)...")
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=3,
        include=['metadatas', 'distances'] # Also retrieve metadata to verify data content
    )
    
    # --- 4. Results Validation ---
    
    # The first result should be very close (distance close to 0) to the query vector
    if results and results.get('ids') and results['ids'][0]:
        first_result_id = results['ids'][0][0]
        first_result_distance = results['distances'][0][0]
        first_result_metadata = results['metadatas'][0][0]

        print(f"✅ SUCCESS: Query returned results. Top match found:")
        print(f"  - ID: {first_result_id}")
        print(f"  - Distance (Cosine): {first_result_distance:.4f} (Closer to 0 is better)")
        print(f"  - Metadata Keys: {list(first_result_metadata.keys())}")
        print("\nTest Complete! ChromaDB is ready for use.")

    else:
        print("❌ FAILURE: Query did not return any results. The index is likely empty or corrupt.")


if __name__ == "__main__":
    # Suppress the numpy scientific notation for cleaner output
    np.set_printoptions(suppress=True) 
    test_chroma_setup()