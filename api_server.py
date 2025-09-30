from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import io
import json
import os
import base64
import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, List, Optional
import uvicorn

# Security
security = HTTPBearer()

# API Key Configuration
API_KEY = os.getenv("DOG_CLASSIFIER_API_KEY", "77ffda4c") 

app = FastAPI(title="Dog Breed Classifier API", version="1.0.0")

# CORS middleware for mobile app requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and classes
model = None
device = None
transform = None
classes = None

# Verify API key from Authorization header
def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Alternative: Verify API key from X-API-Key header
def verify_api_key_header(x_api_key: Optional[str] = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key
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

# Load the trained model and class names
def load_model_and_classes():
    global model, device, transform, classes
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model architecture
    model = models.resnet101(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 120)  # 120 dog breeds
    
    # Load trained weights
    model.load_state_dict(torch.load('./best_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    # Define the same transform used during training
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load class names in the same order as during training
    try:
        # Create a temporary dataset to get the class names in correct order
        temp_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        temp_dataset = ImageFolder(root='./images/Images', transform=temp_transform)
        classes = temp_dataset.classes
        print(f" Loaded {len(classes)} classes from dataset in correct order")
        print(f"First few classes: {classes[:5]}")
        print(f"Last few classes: {classes[-5:]}")
    except Exception as e:
        print(f" Error loading classes from dataset: {e}")
        print(" Falling back to hardcoded class list (may cause prediction errors)")
        # Fallback to hardcoded list
        classes = [
            "Chihuahua", "Japanese spaniel", "Maltese dog", "Pekinese", "Shih-Tzu",
            "Blenheim spaniel", "papillon", "toy terrier", "Rhodesian ridgeback", "Afghan hound",
            "basset", "beagle", "bloodhound", "bluetick", "black-and-tan coonhound",
            "Walker hound", "English foxhound", "redbone", "borzoi", "Irish wolfhound",
            "Italian greyhound", "whippet", "Ibizan hound", "Norwegian elkhound", "otterhound",
            "Saluki", "Scottish deerhound", "Weimaraner", "Staffordshire bullterrier", "American Staffordshire terrier",
            "Bedlington terrier", "Border terrier", "Kerry blue terrier", "Irish terrier", "Norfolk terrier",
            "Norwich terrier", "Yorkshire terrier", "wire-haired fox terrier", "Lakeland terrier", "Sealyham terrier",
            "Airedale", "cairn", "Australian terrier", "Dandie Dinmont", "Boston bull",
            "miniature schnauzer", "giant schnauzer", "standard schnauzer", "Scotch terrier", "Tibetan terrier",
            "silky terrier", "soft-coated wheaten terrier", "West Highland white terrier", "Lhasa", "flat-coated retriever",
            "curly-coated retriever", "golden retriever", "Labrador retriever", "Chesapeake Bay retriever", "German short-haired pointer",
            "vizsla", "English setter", "Irish setter", "Gordon setter", "Brittany spaniel",
            "clumber", "English springer", "Welsh springer spaniel", "cocker spaniel", "Sussex spaniel",
            "Irish water spaniel", "kuvasz", "schipperke", "groenendael", "malinois",
            "briard", "kelpie", "komondor", "Old English sheepdog", "Shetland sheepdog",
            "collie", "Border collie", "Bouvier des Flandres", "Rottweiler", "German shepherd",
            "Doberman", "miniature pinscher", "Greater Swiss Mountain dog", "Bernese mountain dog", "Appenzeller",
            "EntleBucher", "boxer", "bull mastiff", "Tibetan mastiff", "French bulldog",
            "Great Dane", "Saint Bernard", "Eskimo dog", "malamute", "Siberian husky",
            "affenpinscher", "basenji", "pug", "Leonberg", "Newfoundland",
            "Great Pyrenees", "Samoyed", "Pomeranian", "chow", "keeshond",
            "Brabancon griffon", "Pembroke", "Cardigan", "toy poodle", "miniature poodle",
            "standard poodle", "Mexican hairless", "dingo", "dhole", "African hunting dog"
        ]

# Load model when the server starts
@app.on_event("startup")
async def startup_event():
    load_model_and_classes()
    print("Model loaded successfully!")

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Dog Breed Classifier API is running!"}


# Get all available dog breed classes
@app.get("/classes")
async def get_classes():
    return {"classes": classes, "total_classes": len(classes)}

# Debug endpoint to check class order and indices
@app.get("/debug/classes")
async def debug_classes():
    if classes is None:
        return {"error": "Classes not loaded yet"}
    
    debug_info = {
        "total_classes": len(classes),
        "first_10_classes": [(i, classes[i]) for i in range(min(10, len(classes)))],
        "last_10_classes": [(i, classes[i]) for i in range(max(0, len(classes)-10), len(classes))],
        "sample_indices": {
            "affenpinscher": classes.index("affenpinscher") if "affenpinscher" in classes else "not found",
            "yorkshire_terrier": classes.index("yorkshire_terrier") if "yorkshire_terrier" in classes else "not found",
            "Yorkshire_terrier": classes.index("Yorkshire_terrier") if "Yorkshire_terrier" in classes else "not found"
        }
    }
    return debug_info

# Test endpoint to check AKC information fetching for a specific breed
@app.get("/debug/akc/{breed_name}")
async def debug_akc_info(breed_name: str):
    
    breed_info = get_akc_breed_info(breed_name)
    return {
        "breed_name": breed_name,
        "about_breed": breed_info["about_breed"],
        "akc_url": breed_info["akc_url"],
        "success": breed_info["success"],
        "timestamp": "2025-07-16"
    }

# Predict dog breed from uploaded image - Requires API key
@app.post("/predict")
async def predict_breed(file: UploadFile = File(...), api_key: str = Depends(verify_api_key_header)):
    try:
        # Validate file type
        if file.content_type and not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Additional validation - check if it's a valid image by trying to open it
        image_data = await file.read()
        try:
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Apply transforms
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            # Get top 5 predictions
            top5_prob, top5_idx = torch.topk(probabilities, 5, dim=1)
            
            top5_predictions = []
            for i in range(5):
                breed = classes[top5_idx[0][i].item()]
                prob = top5_prob[0][i].item()
                top5_predictions.append({
                    "breed": breed,
                    "confidence": round(prob * 100, 2)
                })
        
        return {
            "success": True,
            "predicted_breed": classes[predicted_idx.item()],
            "confidence": round(confidence.item() * 100, 2),
            "top_5_predictions": top5_predictions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
# Predict dog breeds for multiple images
@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    for file in files:
        try:
            # Process each image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
            results.append({
                "filename": file.filename,
                "predicted_breed": classes[predicted_idx.item()],
                "confidence": round(confidence.item() * 100, 2)
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results}

# Identify dog breed from uploaded image - Google Gemini compatible endpoint
@app.post("/api/identify")
async def identify_breed(file: UploadFile = File(...), api_key: str = Depends(verify_api_key_header)):
    try:
        # Validate file type
        if file.content_type and not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Additional validation
        image_data = await file.read()
        try:
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Apply transforms
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            # Get top 5 predictions
            top5_prob, top5_idx = torch.topk(probabilities, 5, dim=1)
            
            top5_predictions = []
            for i in range(5):
                breed = classes[top5_idx[0][i].item()]
                prob = top5_prob[0][i].item()
                top5_predictions.append({
                    "name": breed,
                    "confidence": round(prob * 100, 2)
                })
        
        return {
            "status": "success",
            "result": {
                "breed": classes[predicted_idx.item()],
                "confidence": round(confidence.item() * 100, 2),
                "alternatives": top5_predictions
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Identification failed: {str(e)}")
    
# Identify animal from base64 image data - Google Gemini app compatible endpoint
@app.post("/identify")
async def identify_animal_gemini(request: dict, api_key: str = Depends(verify_api_key_header)):
    try:
        # Extract image data from request
        if "imageDataUrl" not in request:
            raise HTTPException(status_code=400, detail="Missing imageDataUrl in request")
        
        image_data_url = request["imageDataUrl"]
        
        # Parse base64 image data
        if not image_data_url.startswith("data:image/"):
            raise HTTPException(status_code=400, detail="Invalid image data URL format")
        
        # Extract base64 data
        try:
            header, base64_data = image_data_url.split(",", 1)
            image_data = base64.b64decode(base64_data)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
        
        # Process image
        try:
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Apply transforms
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_breed = classes[predicted_idx.item()]
            confidence_score = confidence.item() * 100
        
        # Fetch real breed information from AKC
        print(f"Fetching AKC info for: {predicted_breed}")
        breed_info = get_akc_breed_info(predicted_breed)
        
        # Return in AnimalInfo format with real AKC data
        return {
            "animalName": predicted_breed,
            "description": breed_info["about_breed"],  # Standard field name
            "aboutThisDog": breed_info["about_breed"],  # Alternative field name
            "confidence": f"Identified with {confidence_score:.1f}% confidence",
            "learnMoreUrl": breed_info["akc_url"],
            "funFacts": [f"This breed was identified with {confidence_score:.1f}% confidence."],
            "sources": [
                {
                    "title": f"{predicted_breed} - American Kennel Club",
                    "url": breed_info["akc_url"],
                    "description": f"Official AKC information about the {predicted_breed} breed"
                },
                {
                    "title": "Dog Breed Classification",
                    "url": "https://en.wikipedia.org/wiki/Dog_breed",
                    "description": "Learn more about dog breeds and their characteristics"
                }
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Identification failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
