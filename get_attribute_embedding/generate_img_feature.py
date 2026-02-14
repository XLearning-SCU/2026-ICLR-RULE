"""Generate image features using CLIP model."""
import os
import json
import argparse
import pickle

import torch
import clip
from PIL import Image


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load existing features if available
    if os.path.exists(args.output_path):
        with open(args.output_path, "rb") as f:
            features_dict = pickle.load(f)
    else:
        features_dict = {}

    processed_count = 0
    for item in data:
        entity_id = item["Entity ID"]
        image_path = os.path.join(args.img_folder, os.path.basename(item["image_id"]))

        if entity_id in features_dict:
            continue

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        try:
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue

        with torch.no_grad():
            image_features = model.encode_image(image).cpu().numpy().squeeze()

        features_dict[entity_id] = image_features
        processed_count += 1

        if processed_count % 100 == 0:
            print(f"Processed {processed_count} entities.")

    with open(args.output_path, "wb") as f:
        pickle.dump(features_dict, f)

    print(f"Complete. Total: {processed_count}. Saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image features using CLIP")
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--img_folder", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
