# Multimodal E-commerce Retrieval System

End-to-end pipeline for the take-home assignment: scrape products from `laam.pk`, create multimodal (text + image) CLIP embeddings, store them in ChromaDB, and power text-based retrieval demos.

## Repository Structure

- `scraper.py` – sitemap crawl + product API ingestion + image downloads (≥1,000 items).
- `embed.py` – loads CLIP via the `llm` library, embeds text and images, concatenates + normalizes vectors, saves `products_metadata_with_embeddings.json`.
- `ingest_to_chroma.py` – reads fused vectors, sanitizes metadata, ingests into persistent Chroma collection `products`.
- `chromadb_test.py` – sanity checks (document count, sample query).
- `main.py` – CLI search driver (single query or interactive loop) that prints query vectors and top matches.
- `demo_queries.py` – runs the three required assignment queries and writes the transcript to `demo_queries_output.txt`.
- `products/` – stored metadata files and downloaded images.

## Prerequisites

- Python 3.10+
- `pip`, `venv`, and system packages needed for PIL/requests (e.g., libjpeg, zlib)
- CLIP model installed for `llm`: `llm install llm-clip`

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> `requirements.txt` is generated from the project virtual environment (`pip freeze > requirements.txt`).

## Running the Pipeline

> **No-scrape option:** If you prefer not to run the scraper, you can start from the ready-made JSON files under `products/`. Download the companion image bundle from [Google Drive](https://drive.google.com/file/d/1iZMoeP8kWvZ6oBRbllZDkxyYD6-uPCMM/view) and extract its `images/` directory directly into `products/`.

1. **Scrape products (text + images)**
   ```bash
   source .venv/bin/activate
   python scraper.py
   ```
   - Outputs `products/products_metadata.json` and downloads images under `products/images/`.

> **Shortcut:** This repo already includes `products/products_metadata_with_embeddings.json`, so you can begin at step 3 to initialize ChromaDB immediately. If you wish to regenerate embeddings, follow the step 2.

2. **Generate multimodal embeddings**
   ```bash
   python embed.py
   ```
  - Reads `products/products_metadata.json` (adjust `PRODUCTS_FILE` if needed).
  - Writes `products/products_metadata_with_embeddings.json` with `{id, vector, metadata}`.

3. **Ingest into ChromaDB**
   ```bash
   python ingest_to_chroma.py
   ```
   - Creates/updates persistent collection `products` at `./chroma_db`.

4. **Ad-hoc search**
   ```bash
   # Single query
   python main.py --query "royal blue sharara suit" --top-k 3

   # Interactive loop
   python main.py
   ```
   - Prints the 1024-d query vector and the top-k matches (IDs, titles, captions, images, distances).

5. **Assignment demo transcript**
   ```bash
   python demo_queries.py --output demo_queries_output.txt
   ```
   - Runs the three mandated queries and saves the results for documentation.

## Demonstration Queries

All runs executed via `python demo_queries.py --output demo_queries_output.txt` on the ingested dataset. Each query prints the computed vector (first few values shown) and top three retrieved items.

### 1. Color & Style – `royal blue sharara suit`
- Vector (dim 1024): `[0.007923, 0.001950, 0.009889, …, 0.052860, -0.019969, 0.018081]`
- Top IDs:
  1. `namaz-chadar-with-sleeves-dark-blue-hue` – Namaz Chadar With Sleeves - Dark Blue Hue (distance 0.3561)  
     <img src="https://cdn.shopify.com/s/files/1/2337/7003/files/media_image-baf10679a0e8415fb543e204a20439cb_16dabbfd-3dcb-479a-8e19-d613ef511daf.jpg?v=1762259586" alt="Namaz Chadar With Sleeves - Dark Blue Hue" width="240" />  
     _Relevance:_ Navy-colored piece matches the explicit “blue” color request.
  2. `falaksar-rkpvus044-navy-blu` – FALAKSAR-RKPVUS044-NAVY-BLU (distance 0.3570)  
     <img src="https://cdn.shopify.com/s/files/1/2337/7003/files/3000000034323_1_-174725775063049_7c3a51b5-4680-470a-af1d-7ba82ff9467a.jpg?v=1747292742" alt="FALAKSAR-RKPVUS044-NAVY-BLU" width="240" />  
     _Relevance:_ Semantic overlap between “royal blue” and “navy” drives the match.
  3. `navy-blue-shalwar-kameez-2` – Navy Blue Shalwar Kameez (distance 0.3571)  
     <img src="https://cdn.shopify.com/s/files/1/2337/7003/files/media_image-dc34c03e428e41389ea823f4e6e70fec_72e6bc3e-2d84-43ae-b702-690b08585a40.jpg?v=1751525202" alt="Navy Blue Shalwar Kameez" width="240" />  
     _Relevance:_ Similar reasoning—Semantic overlap between “royal blue” and “navy” drives the match.

### 2. Fabric / Occasion – `formal wear for a mehndi`
- Vector: `[0.013285, -0.002407, -0.008970, …, 0.045068, 0.013402, -0.002323]`
- Top IDs:
  1. `haze-mehndi-2pc` – Haze mehndi 2pc (distance 0.3298)  
     <img src="https://cdn.shopify.com/s/files/1/2337/7003/files/media_image-ab51be3419764fcab79849dcb264c42d_258fbd53-b122-4da9-8a98-d6829b2bdc93.jpg?v=1762855326" alt="Haze mehndi 2pc" width="240" />  
     _Relevance:_ “Mehndi” keyword aligns directly with the query intent.
  2. `sherwani-004` – sherwani 004 (distance 0.3510)  
     <img src="https://cdn.shopify.com/s/files/1/2337/7003/products/8e92741ec635df26b62fd5ad01798a9b.jpg?v=1701775242" alt="sherwani 004" width="240" />  
     _Relevance:_ Sherwani is semantically tied to wedding/Mehndi occasions.
  3. `chunri-mehroon-digital-print-with-farshi-shalwar` – Chunri Mehroon (Digital Print with Farshi Shalwar) (distance 0.3516)  
     <img src="https://cdn.shopify.com/s/files/1/2337/7003/files/media_image-736294f3e86749d68eca6ad6d19bd17d_a0feda72-5be0-4f00-9c3c-251dc036ad96.jpg?v=1762499734" alt="Chunri Mehroon (Digital Print with Farshi Shalwar)" width="240" />  
     _Relevance:_ “Chunri” signifies traditional wear, fitting the Mehndi vibe.

### 3. Cross-modal imagery – `dress that looks like a flower garden`
- Vector: `[0.007293, -0.011943, 0.010674, …, 0.010357, -0.007080, -0.006228]`
- Top IDs:
  1. `01-piece-ready-to-wear-printed-long-frock-2` – 01 Piece Ready to wear Printed Long frock (distance 0.4010)  
     <img src="https://cdn.shopify.com/s/files/1/2337/7003/files/TH_03222.jpg?v=1757256318" alt="01 Piece Ready to wear Printed Long frock" width="240" />  
     _Relevance:_ Dress is covered in floral prints, matching the “flower garden” imagery.
  2. `floral-contrast-2pcs` – FLORAL CONTRAST 2PCS (distance 0.4061)  
     <img src="https://cdn.shopify.com/s/files/1/2337/7003/files/media_image-5572d9bd41824c9b99419c59ebcc82c5_0667decd-7f51-4538-959c-c5f8fbdc944b.jpg?v=1758263220" alt="FLORAL CONTRAST 2PCS" width="240" />  
     _Relevance:_ “Floral” in the product name reinforces semantic similarity.
  3. `zk-floral-set` – Zk| FLORAL SET (distance 0.4121)  
     <img src="https://cdn.shopify.com/s/files/1/2337/7003/files/media_image-e81d54b07f2f44fcb77a0bee71cc91eb_435569af-b0e1-43d7-9e39-79e425b628ae.jpg?v=1758778265" alt="Zk| FLORAL SET" width="240" />  
     _Relevance:_ “Floral” naming again cues the requested flowery aesthetic.

Complete transcripts (with captions and image paths) live in `demo_queries_output.txt`.

## Notes

- **Chroma persistence**: All scripts assume the database lives in `./chroma_db`. Delete that folder to start fresh.
- **Vector dimension**: Stored embeddings are 1024-d (text+image concatenation). Query vectors repeat the 512-d CLIP text embedding until they reach 1024 dims, then normalize.


