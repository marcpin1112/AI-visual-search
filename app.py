# Install: pip install flask torch torchvision openai clip-by-openai
import clip
import torch
from flask import Flask, request, jsonify
from PIL import Image
import requests
import io

app = Flask(__name__)
model, preprocess = clip.load("ViT-B/32")

# Mock product dataset
products = [
    {"title": "Blue Hoodie", "url": "https://yourstore.com/products/blue-hoodie", "image": "https://yourcdn.com/blue-hoodie.jpg"},
    {"title": "Red Jacket", "url": "https://yourstore.com/products/red-jacket", "image": "https://yourcdn.com/red-jacket.jpg"},
    # Add more with image URLs
]

def encode_image(url):
    response = requests.get(url)
    image = preprocess(Image.open(io.BytesIO(response.content))).unsqueeze(0)
    with torch.no_grad():
        return model.encode_image(image)

# Precompute product image vectors
product_features = [encode_image(p["image"]) for p in products]

@app.route("/search", methods=["POST"])
def search():
    image = Image.open(request.files["image"])
    image_input = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        image_feature = model.encode_image(image_input)

    scores = [torch.cosine_similarity(image_feature, f)[0].item() for f in product_features]
    top_products = sorted(zip(scores, products), reverse=True)[:5]
    return jsonify([p for _, p in top_products])

if __name__ == "__main__":
    app.run(debug=True)