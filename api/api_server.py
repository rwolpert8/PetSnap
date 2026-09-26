"""PetSnap API and browser demo. Run from the repository root with uvicorn."""
from contextlib import asynccontextmanager
from typing import List, Optional
import base64
import binascii
import os
import secrets

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

if __package__:
    from .inference import BASE_DIR, MAX_UPLOAD_BYTES, Classifier, InvalidImage, InferenceBusy
else:  # Also support python api/api_server.py.
    from inference import BASE_DIR, MAX_UPLOAD_BYTES, Classifier, InvalidImage, InferenceBusy

API_KEY = os.getenv("DOG_CLASSIFIER_API_KEY")

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier = await run_in_threadpool(Classifier)
    yield
    del app.state.classifier

app = FastAPI(title="PetSnap", version="2.0.0", lifespan=lifespan)
origins = [origin.strip() for origin in os.getenv("PETSNAP_CORS_ORIGINS", "").split(",") if origin.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins,
                       allow_methods=["GET", "POST"], allow_headers=["X-API-Key", "Content-Type"])

def verify_api_key_header(x_api_key: Optional[str] = Header(None)):
    if not API_KEY or not secrets.compare_digest(x_api_key or "", API_KEY):
        raise HTTPException(status_code=401, detail="A configured API key is required.")

async def read_upload(file: UploadFile):
    try:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Choose an image smaller than 10 MB.")
        return data
    finally:
        await file.close()

async def run_prediction(data: bytes):
    try:
        return await run_in_threadpool(app.state.classifier.predict, data)
    except InvalidImage as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except InferenceBusy as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "2"}) from exc

# Fetch breed information from AKC website
def get_akc_breed_info(breed_name):
    try:
        # Clean up breed name for URL
        breed_url_name = breed_name.lower().replace(" ", "-").replace("_", "-")
        
        # Handle special cases for AKC URLs
        breed_mappings = {
            "shih-tzu": "shih-tzu",
            "wire-haired-fox-terrier": "wire-fox-terrier",
            "soft-coated-wheaten-terrier": "soft-coated-wheaten-terrier",
            "west-highland-white-terrier": "west-highland-white-terrier",
            "flat-coated-retriever": "flat-coated-retriever",
            "curly-coated-retriever": "curly-coated-retriever",
            "chesapeake-bay-retriever": "chesapeake-bay-retriever",
            "german-short-haired-pointer": "german-shorthaired-pointer",
            "old-english-sheepdog": "old-english-sheepdog",
            "border-collie": "border-collie",
            "bouvier-des-flandres": "bouvier-des-flandres",
            "greater-swiss-mountain-dog": "greater-swiss-mountain-dog",
            "bernese-mountain-dog": "bernese-mountain-dog",
            "bull-mastiff": "bullmastiff",
            "tibetan-mastiff": "tibetan-mastiff",
            "french-bulldog": "french-bulldog",
            "great-dane": "great-dane",
            "saint-bernard": "saint-bernard",
            "eskimo-dog": "american-eskimo-dog",
            "siberian-husky": "siberian-husky",
            "great-pyrenees": "great-pyrenees",
            "brabancon-griffon": "brussels-griffon",
            "toy-poodle": "poodle-toy",
            "miniature-poodle": "poodle-miniature", 
            "standard-poodle": "poodle-standard",
            "mexican-hairless": "xoloitzcuintli",
            "african-hunting-dog": "african-wild-dog"
        }
        
        if breed_url_name in breed_mappings:
            breed_url_name = breed_mappings[breed_url_name]
        
        akc_url = f"https://www.akc.org/dog-breeds/{breed_url_name}/"
        
        # Enhanced headers to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        print(f"Attempting to fetch: {akc_url}")
        response = requests.get(akc_url, headers=headers, timeout=15)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Debug: Print page title
            page_title = soup.find('title')
            print(f"Page title: {page_title.get_text().strip() if page_title else 'No title found'}")
            
            about_section = ""
            
            # Strategy 1: Look for specific patterns that contain breed descriptions
            page_content = response.text
            
            # Look for the share modal content which contains clean breed descriptions
            if 'share-modal__content-inner' in page_content:
                try:
                    # Find the content between share-modal__content-inner tags
                    start_pattern = '<div class="share-modal__content-inner mt3">'
                    start_idx = page_content.find(start_pattern)
                    if start_idx != -1:
                        start_idx += len(start_pattern)
                        end_idx = page_content.find('</div>', start_idx)
                        if end_idx != -1:
                            modal_content = page_content[start_idx:end_idx].strip()
                            # Clean up HTML and get text
                            modal_soup = BeautifulSoup(modal_content, 'html.parser')
                            clean_text = modal_soup.get_text().strip()
                            if len(clean_text) > 100 and 'breed' in clean_text.lower():
                                about_section = clean_text
                                print(f"Found description in share modal: {about_section[:100]}...")
                except Exception as e:
                    print(f"Error extracting share modal content: {e}")
            
            # Strategy 2: Look for JSON data in script tags
            if not about_section:
                script_tags = soup.find_all('script', string=True)
                for script in script_tags:
                    script_content = script.string
                    if script_content and '"post_content":"' in script_content:
                        try:
                            # Extract content between post_content quotes
                            start = script_content.find('"post_content":"') + len('"post_content":"')
                            end = script_content.find('","post_title"', start)
                            if end == -1:
                                end = script_content.find('","post_excerpt"', start)
                            if end == -1:
                                end = script_content.find('",', start)
                            
                            if end > start:
                                description = script_content[start:end]
                                # Clean up escaped characters
                                description = description.replace('\\u201c', '"').replace('\\u201d', '"')
                                description = description.replace('\\u2019', "'").replace('\\u2014', "—")
                                description = description.replace('\\"', '"').replace('\\/', '/')
                                
                                if len(description) > 100 and any(word in description.lower() for word in ['breed', 'dog']):
                                    about_section = description
                                    print(f"Found breed description in JSON: {about_section[:100]}...")
                                    break
                                    
                        except Exception as e:
                            print(f"Error parsing JSON in script: {e}")
                            continue
            
            # Strategy 2: Look for share modal content (backup approach)
            if not about_section:
                share_content = soup.select('.share-modal__content-inner')
                if share_content:
                    text = share_content[0].get_text().strip()
                    if len(text) > 100:
                        about_section = text
                        print(f"Found content in share modal: {about_section[:100]}...")
            
            # Strategy 3: Traditional paragraph scraping (fallback)
            if not about_section:
                strategies = [
                    lambda: soup.select('div[class*="breed-hero"] p, div[class*="breed-overview"] p'),
                    lambda: soup.select('main p, .main-content p, .content p'),
                    lambda: soup.select('article p, .article p'),
                    lambda: soup.find_all('p'),
                ]
                
                for i, strategy in enumerate(strategies):
                    try:
                        paragraphs = strategy()
                        print(f"Fallback strategy {i+1}: Found {len(paragraphs)} paragraphs")
                        
                        if paragraphs:
                            good_paragraphs = []
                            for p in paragraphs:
                                text = p.get_text().strip()
                                
                                if (len(text) < 50 or 
                                    any(skip in text.lower() for skip in [
                                        'cookie', 'privacy', 'newsletter', 'subscribe', 
                                        'follow us', 'social media', 'advertisement',
                                        'terms of service', 'contact', 'email',
                                        'javascript', 'browser', 'enable'
                                    ])):
                                    continue
                                
                                if any(word in text.lower() for word in [
                                    'breed', 'dog', 'originally', 'developed', 'known for',
                                    'temperament', 'character', 'history', 'bred', 'companion',
                                    'working', 'sporting', 'hunting', 'size', 'coat', 'training'
                                ]):
                                    good_paragraphs.append(text)
                                    if len(' '.join(good_paragraphs)) > 400:
                                        break
                            
                            if good_paragraphs and len(' '.join(good_paragraphs)) > 100:
                                about_section = ' '.join(good_paragraphs[:3])
                                print(f"Fallback strategy {i+1} succeeded")
                                break
                                
                    except Exception as e:
                        print(f"Fallback strategy {i+1} failed: {e}")
                        continue
            
            # Clean up the content
            if about_section:
                # Remove extra whitespace
                about_section = re.sub(r'\s+', ' ', about_section).strip()
                # Limit length
                if len(about_section) > 800:
                    about_section = about_section[:800] + "..."
                print(f"Final content length: {len(about_section)}")
            else:
                print("No content found with any strategy")
            
            # If we found content, return it
            if about_section and len(about_section) > 100:
                return {
                    "about_breed": about_section,
                    "akc_url": akc_url,
                    "success": True
                }
        
        # If we get here, scraping failed
        raise Exception(f"Could not extract content from {akc_url}")
            
    except Exception as e:
        print(f"Error fetching AKC info for {breed_name}: {e}")
        
        # Return fallback information
        return {
            "about_breed": f"No AKC information found for {breed_name}. Please check the breed name or try again later.",
            "akc_url": f"https://www.akc.org/dog-breeds/{breed_name.lower().replace(' ', '-').replace('_', '-')}/",
            "success": False
        }


@app.get("/health")
async def health():
    return {"status": "ready", "classes": len(app.state.classifier.classes)}

@app.get("/classes")
async def get_classes():
    return {"classes": app.state.classifier.classes, "total_classes": len(app.state.classifier.classes)}

@app.post("/api/demo/predict")
async def demo_predict(file: UploadFile = File(...)):
    # Intentionally public: browser clients must never contain a secret API key.
    return await run_prediction(await read_upload(file))

@app.post("/predict", dependencies=[Depends(verify_api_key_header)])
async def predict_breed(file: UploadFile = File(...)):
    return await run_prediction(await read_upload(file))

@app.post("/predict_batch", dependencies=[Depends(verify_api_key_header)])
async def predict_batch(files: List[UploadFile] = File(...)):
    if len(files) > 10:
        for file in files:
            await file.close()
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    results = []
    try:
        for file in files:
            filename = file.filename
            try:
                prediction = await run_prediction(await read_upload(file))
                results.append({"filename": filename, "predicted_breed": prediction["predicted_breed"],
                                "confidence": prediction["confidence"]})
            except HTTPException as exc:
                results.append({"filename": filename, "error": exc.detail})
    finally:
        for file in files:
            await file.close()
    return {"results": results}

@app.post("/api/identify", dependencies=[Depends(verify_api_key_header)])
async def identify_breed(file: UploadFile = File(...)):
    prediction = await run_prediction(await read_upload(file))
    return {"status": "success", "result": {"breed": prediction["predicted_breed"],
            "confidence": prediction["confidence"], "alternatives": [
                {"name": item["breed"], "confidence": item["confidence"]}
                for item in prediction["top_5_predictions"]]}}

@app.post("/identify", dependencies=[Depends(verify_api_key_header)])
async def identify_animal(request: dict):
    value = request.get("imageDataUrl")
    if not isinstance(value, str) or not value.startswith("data:image/") or "," not in value:
        raise HTTPException(status_code=400, detail="A valid imageDataUrl is required")
    if len(value) > (MAX_UPLOAD_BYTES * 4 // 3) + 1024:
        raise HTTPException(status_code=413, detail="Choose an image smaller than 10 MB.")
    try:
        data = base64.b64decode(value.split(",", 1)[1], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image data") from exc
    prediction = await run_prediction(data)
    breed = prediction["predicted_breed"]
    info = await run_in_threadpool(get_akc_breed_info, breed)
    return {"animalName": breed, "description": info["about_breed"],
            "aboutThisDog": info["about_breed"],
            "confidence": f"Identified with {prediction['confidence']:.1f}% confidence",
            "learnMoreUrl": info["akc_url"], "funFacts": [],
            "sources": [{"title": f"{breed} - American Kennel Club", "url": info["akc_url"]}]}

# Keep API routes ahead of the static app. Nothing in models/ or Images/ is exposed.
app.mount("/", StaticFiles(directory=BASE_DIR / "frontend", html=True), name="web")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
