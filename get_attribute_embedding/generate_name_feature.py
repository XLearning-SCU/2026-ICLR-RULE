"""Generate entity name features using CLIP model."""
import os
import json
import argparse
import pickle
import re

import torch
import clip


def pre_caption(caption, max_words=50):
    """Preprocess caption text."""
    caption = re.sub(r"([.!\"()*#:;~])", ' ', caption.lower())
    caption = re.sub(r"\s{2,}", ' ', caption)
    caption = caption.rstrip('\n').strip(' ')
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-L/14", device=device)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if os.path.exists(args.output_path):
        with open(args.output_path, "rb") as f:
            features_dict = pickle.load(f)
    else:
        features_dict = {}

    processed_count = 0
    for item in data:
        entity_id = item[0]
        name_list = item[1]

        if entity_id in features_dict:
            continue

        try:
            caption = " ".join(name_list)
            caption = pre_caption(caption, max_words=77)
            text_tokens = clip.tokenize(caption, truncate=True).to(device)
            
            with torch.no_grad():
                text_features = model.encode_text(text_tokens).cpu().numpy().squeeze()

            features_dict[entity_id] = text_features
            processed_count += 1

            if processed_count % 100 == 0:
                print(f"Processed {processed_count} entity names.")

        except Exception as e:
            print(f"Error processing Entity ID {entity_id}: {e}")
            continue

    with open(args.output_path, "wb") as f:
        pickle.dump(features_dict, f)

    print(f"Complete. Total: {processed_count}. Saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate name features using CLIP")
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
