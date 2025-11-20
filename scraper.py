import requests
import xml.etree.ElementTree as ET
import json
import os
import time
import re
from bs4 import BeautifulSoup
from functools import wraps

# --- CONFIGURATION ---
PRODUCTS_DIR = "products"
IMAGES_DIR = os.path.join(PRODUCTS_DIR, "images")
SITEMAP_URL = "https://laam.pk/products-sitemap.xml"
API_GATEWAY = "https://gateway.laam.pk/v1/products"
MAX_RETRIES = 3
INITIAL_BACKOFF = 2 # seconds

# Key headers to mimic a browser for the API requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:142.0) Gecko/20100101 Firefox/142.0",
    "Accept": "application/json, text/plain, */*",
    "store-identifier": "laam.pk",
    "Origin": "https://laam.pk",
    "Sec-Fetch-Mode": "cors"
}
# --- END CONFIGURATION ---

# Ensure directories exist
os.makedirs(IMAGES_DIR, exist_ok=True)

# --- UTILITY: RETRY DECORATOR ---

def retry_request(max_retries=MAX_RETRIES, initial_backoff=INITIAL_BACKOFF):
    """Decorator to retry a function (network request) on failure with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    # Retry on connection errors, timeouts, or specific HTTP status errors (5xx, 429)
                    if attempt < max_retries - 1:
                        sleep_time = initial_backoff * (2 ** attempt) + (0.1 * attempt)
                        print(f"  [Retry {attempt+1}/{max_retries}] Request failed ({e}). Retrying in {sleep_time:.2f}s...")
                        time.sleep(sleep_time)
                    else:
                        raise # Re-raise the exception after the final attempt
        return wrapper
    return decorator

# --- CORE UTILITY FUNCTIONS ---

def clean_html(html_string):
    """Cleans HTML tags and excessive whitespace from the API description."""
    if not html_string:
        return ""
    
    soup = BeautifulSoup(html_string, 'html.parser')
    clean_text = soup.get_text(separator=' ', strip=True)
    
    clean_text = clean_text.replace('\u200b', '')
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    return clean_text

@retry_request()
def fetch_sitemap_content():
    """Fetches the sitemap XML with retry logic."""
    response = requests.get(SITEMAP_URL, timeout=10)
    response.raise_for_status() # Will raise HTTPError for 4xx/5xx responses
    return response.text

def parse_sitemap(xml_content):
    """Parses the sitemap to extract the product URL (handle) AND caption."""
    products = []
    root = ET.fromstring(xml_content)
    ns = {'s': 'http://www.sitemaps.org/schemas/sitemap/0.9', 'image': 'http://www.google.com/schemas/sitemap-image/1.1'}
    for url_element in root.findall('s:url', ns):
        loc = url_element.find('s:loc', ns).text
        image_caption_element = url_element.find('image:image/image:caption', ns)
        
        caption = image_caption_element.text if image_caption_element is not None else ""
        
        products.append({
            "product_url": loc,
            "image_caption": caption  
        }) 
    return products

@retry_request()
def fetch_product_details_from_api(handle):
    """Fetches the detailed JSON data for a single product handle with retry logic."""
    url = f"{API_GATEWAY}?handle={handle}"
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status() 
    return response.json()

@retry_request()
def download_image_content(image_url):
    """Downloads image content with retry logic."""
    response = requests.get(image_url, headers=HEADERS, timeout=10, stream=True)
    response.raise_for_status()
    return response.content

# --- MAIN SCRAPING LOGIC ---
def scrape_laam_pk(num_products=1000):
    """
    Main function to crawl sitemap, fetch API details, clean data, and save images.
    """
    # Use the retry-enabled function here
    sitemap_content = None
    try:
        sitemap_content = fetch_sitemap_content()
    except requests.exceptions.RequestException as e:
        print(f"FATAL: Could not fetch sitemap after {MAX_RETRIES} attempts: {e}")
        return

    # Phase 1: Get all product handles and captions from the sitemap
    product_urls = parse_sitemap(sitemap_content)
    products_to_process = product_urls[:num_products]

    final_products_data = []
    
    for i, product_item in enumerate(products_to_process):
        url = product_item.get('product_url', '')
        if '/products/' not in url:
            continue

        try:
            product_handle = url.split('/products/')[1].split('?')[0]
        except IndexError:
            continue 

        sitemap_caption = product_item.get("image_caption", "")
        
        print(f"Processing product {i+1}/{len(products_to_process)}: {product_handle}")
        
        # 2. Fetch Detailed Data from API (Retries built-in)
        api_data = None
        try:
            api_data = fetch_product_details_from_api(product_handle)
        except requests.exceptions.RequestException:
            # Fatal error was logged and retried inside the decorator, now skip the product
            print(f"Skipping {product_handle} due to persistent API retrieval failure.")
            continue
        
        if not api_data or not api_data.get('title'):
            print(f"Skipping {product_handle} due to empty/invalid API data.")
            continue
            
        # 3. Extract and Prepare Data for Vector DB
        product_data = {}
        
        # --- PRIMARY IDENTIFIERS & METADATA ---
        product_data["unique_item_id"] = product_handle
        product_data["product_title"] = api_data.get("title", "")
        product_data["price"] = api_data.get("price", None) 
        product_data["sitemap_caption"] = sitemap_caption 

        size_options = api_data.get("options", [])
        if size_options and size_options[0].get("name") == "Size":
            product_data["size_options"] = size_options[0].get("values", [])
        else:
            product_data["size_options"] = []
            
        attributes = api_data.get("attributed_description")
        if attributes is None:
            attributes = {}
            
        product_data["structured_attributes"] = attributes 
        
        # --- FULL TEXT DESCRIPTION (Optimized for CLIP Encoding) ---
        raw_description = api_data.get("description", "")
        clean_desc = clean_html(raw_description)
        
        attribute_summary = ' '.join([f"{k}: {v['label']}" for k, v in attributes.items()])
        
        product_data["full_text_description"] = (
            f"TITLE: {product_data['product_title']}. "
            f"CAPTION: {sitemap_caption}. "
            f"ATTRIBUTES: {attribute_summary}. " 
            f"DETAILS: {clean_desc}"
        )
        
        # 4. Extract Image URL 
        image_url = api_data.get("media", [{}])[0].get("src")
        product_data["image_url"] = image_url

        # 5. Download Image (Retries built-in)
        if image_url:
            image_name = f"{product_handle}.jpg"
            image_path = os.path.join(IMAGES_DIR, image_name)
            try:
                # Use the retry-enabled download function
                img_data = download_image_content(image_url)
                with open(image_path, 'wb') as handler:
                    handler.write(img_data)
                product_data["local_image_path"] = image_path
            except requests.exceptions.RequestException as e:
                print(f"Skipping image for {product_handle} due to persistent download failure.")
                product_data["local_image_path"] = None
        else:
            product_data["local_image_path"] = None

        final_products_data.append(product_data)
        # NOTE: Removed explicit time.sleep(0.1) as retries provide necessary politeness and delay
        
    # Phase 3: Save Final Metadata
    output_path = os.path.join(PRODUCTS_DIR, "products_metadata.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_products_data, f, ensure_ascii=False, indent=4)
        
    print(f"\n✨ Successfully acquired {len(final_products_data)} products.")
    print(f"Metadata saved to {output_path}")
    
    return final_products_data

if __name__ == "__main__":
    scrape_laam_pk(num_products=1200)