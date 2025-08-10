# clip_image_compare.py
"""
CLIP image similarity utility.

Install dependencies before running:
pip install torch torchvision ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install pillow scikit-learn

Example:
python clip_image_compare.py --img1 check.png --img2 check.jpg
"""

import argparse
import torch
import clip
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

# Load model and preprocessing
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_clip_embedding(image_path):
    """Load image, preprocess and return CLIP image embedding as a numpy array."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Open image and ensure conversion to RGB
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        image_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image_tensor)
        # normalize embedding to unit vector for cosine similarity stability
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()

def compare_images(img1_path, img2_path):
    """Return cosine similarity between two images using CLIP embeddings."""
    emb1 = get_clip_embedding(img1_path)
    emb2 = get_clip_embedding(img2_path)
    similarity = cosine_similarity(emb1, emb2)[0][0]
    return float(similarity)

def main():
    parser = argparse.ArgumentParser(description="Compare two images using OpenAI CLIP (ViT-B/32).")
    parser.add_argument("--img1", required=True, help="Path to first image")
    parser.add_argument("--img2", required=True, help="Path to second image")
    parser.add_argument("--verbose", action="store_true", help="Print extra info")
    args = parser.parse_args()

    try:
        score = compare_images(args.img1, args.img2)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"CLIP Cosine Similarity Score: {score:.6f}")
    if args.verbose:
        print(f"Device used: {device}")
        print(f"Image 1: {args.img1}")
        print(f"Image 2: {args.img2}")

if __name__ == "__main__":
    main()
