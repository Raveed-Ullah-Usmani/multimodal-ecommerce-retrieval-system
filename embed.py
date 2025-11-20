import llm
import json
import numpy as np
import os
from PIL import Image

# --- CONFIGURATION ---
# IMPORTANT: Adjust these paths if your setup differs
PRODUCTS_FILE = os.path.join("products-1", "products_metadata.json")
OUTPUT_FILE = os.path.join("products-1", "products_metadata_with_embeddings.json")

# Initialize the CLIP model globally
try:
    # Use 'clip' as the name if it was installed using 'llm install llm-clip'
    CLIP_MODEL = llm.get_embedding_model("clip") 
except Exception as e:
    print(f"Error initializing CLIP model: {e}")
    print("Please ensure you ran 'llm install llm-clip' in your environment.")
    exit()
# --- END CONFIGURATION ---

# --- EMBEDDING FUNCTIONS ---

def generate_text_embedding(model, text):
    """Generates an embedding for the given text using the llm API."""
    # llm.embed is used to get the vector representation
    return model.embed(text)

def generate_image_embedding(model, image_path):
    """Generates an embedding for the given image."""
    try:
        return model.embed(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}. Skipping image embedding.")
        return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def fuse_embeddings(text_embedding, image_embedding):
    """Fuses text and image embeddings using the Concatenation method."""
    if text_embedding is None or image_embedding is None:
        return None
    
    # Convert lists to NumPy arrays for efficient operation
    text_embedding_np = np.array(text_embedding)
    image_embedding_np = np.array(image_embedding)
    
    # Check for empty vectors (should not happen if embedding succeeded)
    if text_embedding_np.size == 0 or image_embedding_np.size == 0:
         return None

    # FUSION METHOD: Concatenation (The robust choice for retaining all modal data)
    fused_embedding = np.concatenate((text_embedding_np, image_embedding_np))
    
    # Optional: Normalize the fused vector (common practice for vector search)
    fused_embedding /= np.linalg.norm(fused_embedding)
    
    # Convert back to a list of floats for JSON serialization and DB ingestion
    return fused_embedding.tolist()

# --- MAIN PROCESS ---

def process_all_products():
    """Loads metadata, generates, fuses embeddings, and saves in DB-ready format."""
    print("Starting Multimodal Embedding and Fusion Process...")
    
    try:
        with open(PRODUCTS_FILE, 'r', encoding='utf-8') as f:
            products = json.load(f)
    except FileNotFoundError:
        print(f"Error: {PRODUCTS_FILE} not found. Please verify the path.")
        return

    ingestion_data = []
    total_products = len(products)
    
    for i, product in enumerate(products):
        product_id = product.get("unique_item_id")
        
        if i % 100 == 0 and i > 0:
            print(f"Processed {i}/{total_products} products...")

        # --- 1. Generate Embeddings ---
        
        # CRITICAL FIX: Use the rich 'full_text_description'
        text_to_embed = product.get("full_text_description")
        text_embedding = generate_text_embedding(CLIP_MODEL, text_to_embed)
        
        image_path = product.get("local_image_path")
        image_embedding = generate_image_embedding(CLIP_MODEL, image_path)

        if text_embedding is None or image_embedding is None:
             print(f"Skipping {product_id} due to failed embedding generation.")
             continue
        
        # --- 2. Fuse Embeddings ---
        fused_vector = fuse_embeddings(text_embedding, image_embedding)
        
        if fused_vector is None:
            print(f"Skipping {product_id} due to failed vector fusion.")
            continue

        # --- 3. Prepare for Vector DB Ingestion (REQUIRED STRUCTURE) ---
        
        # Keep only the necessary metadata fields for filtering and display
        metadata = {
            "product_title": product.get("product_title"),
            "price": product.get("price"),
            "size_options": product.get("size_options"),
            "sitemap_caption": product.get("sitemap_caption"),
            "full_text_description": product.get("full_text_description"), # Keep source text
            "image_url": product.get("image_url"),
            "local_image_path": product.get("local_image_path"),
        }
        
        # Final structure matching the assignment's Vector DB schema
        ingestion_item = {
            "id": product_id,
            "vector": fused_vector,
            "metadata": metadata
        }
        
        ingestion_data.append(ingestion_item)

    print(f"\nSaving {len(ingestion_data)} fused vectors to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(ingestion_data, f, ensure_ascii=False, indent=2)
    
    print("\n✅ Multimodal Embedding and Fusion Complete!")
    print(f"Output is ready for Vector DB ingestion: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_all_products()