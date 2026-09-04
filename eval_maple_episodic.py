"""
eval_maple_episodic.py
======================
Episodic 5-way 5-shot evaluation for MaPLe (or Zero-shot CLIP ViT-B/16) on Fruit Dataset.

Directly comparable to SWAT and AGNN results.

Protocol:
  1. Load trained MaPLe model (or zero-shot CLIP ViT-B/16)
  2. Extract feature representations for all images in the test split
  3. Run N episodes (default: 600):
       - Sample K=5 support images per class -> compute prototype (mean normalized feature)
       - Sample Q=15 query images per class -> classify by cosine similarity to prototypes
  4. Report Mean Accuracy +/- 95% Confidence Interval
  5. Save JSON summary for cross-model comparison tables

Usage:
  # Evaluate fine-tuned MaPLe
  python eval_maple_episodic.py \
      --model-dir output/base2new/train_base/fruit/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx/seed1 \
      --split-path D:/Fewshot-Fruit/test_split.json \
      --dataset-path D:/Fewshot-Fruit/archive/images/images \
      --n_episodes 600 --n_way 5 --k_shot 5 --n_query 15

  # Baseline: Zero-shot CLIP ViT-B/16
  python eval_maple_episodic.py \
      --zeroshot \
      --split-path D:/Fewshot-Fruit/test_split.json \
      --dataset-path D:/Fewshot-Fruit/archive/images/images \
      --n_episodes 600 --n_way 5 --k_shot 5 --n_query 15
"""

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

_possible_dassl_dirs = [
    "/kaggle/working/Dassl.pytorch",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dassl.pytorch")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "Dassl.pytorch")),
]
for _p in _possible_dassl_dirs:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from clip import clip
from datasets.fruit import CLASS_NAME_MAP, IMG_EXTENSIONS


class FruitTestDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


def build_transform(size=224):
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])


def resolve_split_path(user_split_path, dataset_path):
    if user_split_path and os.path.isfile(user_split_path):
        return os.path.abspath(user_split_path)
    candidates = [
        "D:/Fewshot-Fruit/test_split.json",
        "../test_split.json",
        "../../test_split.json",
        os.path.join(dataset_path or "", "test_split.json"),
        os.path.join(dataset_path or "", "..", "test_split.json"),
        os.path.join(dataset_path or "", "..", "..", "test_split.json"),
        "/kaggle/input/test_split.json",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"Cannot find test_split.json. Please provide --split-path.")


def resolve_dataset_path(user_path, sample_classes):
    if user_path and os.path.isdir(user_path):
        # Check if direct
        if any(os.path.isdir(os.path.join(user_path, c)) for c in sample_classes[:2]):
            return os.path.abspath(user_path)
        for sub in [
            os.path.join("archive", "images", "images"),
            os.path.join("images", "images"),
            "images",
        ]:
            c_dir = os.path.join(user_path, sub)
            if os.path.isdir(c_dir) and any(os.path.isdir(os.path.join(c_dir, c)) for c in sample_classes[:2]):
                return os.path.abspath(c_dir)
        return os.path.abspath(user_path)

    candidates = [
        "D:/Fewshot-Fruit/archive/images/images",
        "/kaggle/input/fruit-recognition/archive/images/images",
        "/kaggle/input/fruit-recognition/images/images",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"Cannot find dataset images directory. Please provide --dataset-path.")


def load_test_images(split_path, dataset_path, partition="test"):
    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    target_classes = sorted(splits.get(partition, []))
    print(f"Loading '{partition}' split ({len(target_classes)} classes): {target_classes}")

    class_to_idx = {c: i for i, c in enumerate(target_classes)}
    image_paths = []
    labels = []

    for c in target_classes:
        cls_dir = os.path.join(dataset_path, c)
        if not os.path.isdir(cls_dir):
            print(f"  WARNING: Directory not found: {cls_dir}")
            continue
        c_imgs = [
            os.path.join(cls_dir, f)
            for f in sorted(os.listdir(cls_dir))
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
        ]
        image_paths.extend(c_imgs)
        labels.extend([class_to_idx[c]] * len(c_imgs))
        print(f"  [{c:25s}] (id={class_to_idx[c]}): {len(c_imgs)} images")

    print(f"Total {len(image_paths)} images across {len(target_classes)} classes.\n")
    return image_paths, labels, target_classes


def extract_features(model, dataloader, device, is_maple=True):
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        if is_maple:
            prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = model.prompt_learner()
            for images, labels in tqdm(dataloader, desc="Extracting MaPLe visual features"):
                images = images.to(device).type(model.dtype)
                feats = model.image_encoder(images, shared_ctx, deep_compound_prompts_vision)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_features.append(feats.float().cpu())
                all_labels.append(labels)
        else:
            for images, labels in tqdm(dataloader, desc="Extracting Zero-shot CLIP features"):
                images = images.to(device)
                feats = model.encode_image(images)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_features.append(feats.float().cpu())
                all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return features, labels


def run_episodic_evaluation(features, labels, n_episodes=600, n_way=5, k_shot=5, n_query=15, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Group features by class index
    class_indices = {}
    unique_classes = torch.unique(labels).tolist()
    for c in unique_classes:
        class_indices[c] = (labels == c).nonzero(as_tuple=True)[0].tolist()

    available_ways = len(unique_classes)
    n_way = min(n_way, available_ways)

    accuracies = []

    for ep in tqdm(range(n_episodes), desc=f"Evaluating {n_episodes} episodes ({n_way}-way {k_shot}-shot)"):
        # Sample n_way classes
        selected_classes = random.sample(unique_classes, n_way)

        support_feats_list = []
        query_feats_list = []
        query_labels_list = []

        for new_label, c in enumerate(selected_classes):
            c_indices = class_indices[c]
            needed = k_shot + n_query
            if len(c_indices) < needed:
                # If class has fewer images, sample with replacement
                sampled = random.sample(c_indices, min(len(c_indices), k_shot))
                # Fill rest
                query_sampled = random.choices(c_indices, k=n_query)
            else:
                chosen = random.sample(c_indices, needed)
                sampled = chosen[:k_shot]
                query_sampled = chosen[k_shot:needed]

            support_f = features[sampled]  # [k_shot, dim]
            proto = support_f.mean(dim=0, keepdim=True)  # [1, dim]
            proto = proto / proto.norm(dim=-1, keepdim=True)
            support_feats_list.append(proto)

            query_f = features[query_sampled]  # [n_query, dim]
            query_feats_list.append(query_f)
            query_labels_list.extend([new_label] * len(query_sampled))

        # Prototypes shape: [n_way, dim]
        prototypes = torch.cat(support_feats_list, dim=0)
        # Queries shape: [n_way * n_query, dim]
        queries = torch.cat(query_feats_list, dim=0)
        true_labels = torch.tensor(query_labels_list, dtype=torch.long)

        # Cosine similarity: [N_queries, n_way]
        sims = queries @ prototypes.t()
        preds = sims.argmax(dim=-1)

        acc = (preds == true_labels).float().mean().item() * 100.0
        accuracies.append(acc)

    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))
    ci95 = float(1.96 * std_acc / np.sqrt(n_episodes))

    return mean_acc, ci95, std_acc


def main():
    parser = argparse.ArgumentParser(description="Episodic Few-Shot Evaluation for MaPLe / CLIP.")
    parser.add_argument("--model-dir", type=str, default="", help="Path to MaPLe model checkpoint dir")
    parser.add_argument("--load-epoch", type=int, default=None, help="Specific epoch to load")
    parser.add_argument("--zeroshot", action="store_true", help="Run Zero-Shot CLIP baseline instead of MaPLe")
    parser.add_argument("--split-path", type=str, default="", help="Path to test_split.json")
    parser.add_argument("--dataset-path", type=str, default="", help="Path to fruit images directory")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"], help="Split partition to evaluate")
    parser.add_argument("--n_episodes", type=int, default=600, help="Number of episodes (default: 600)")
    parser.add_argument("--n_way", type=int, default=5, help="Number of ways (default: 5)")
    parser.add_argument("--k_shot", type=int, default=5, help="Number of shots (default: 5)")
    parser.add_argument("--n_query", type=int, default=15, help="Number of query images per class (default: 15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for episodic sampling")
    parser.add_argument("--feat-cache", type=str, default="", help="Path to save/load precomputed feature cache")
    parser.add_argument("--output-json", type=str, default="", help="Path to save results JSON")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for feature extraction")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Resolve paths
    split_path = resolve_split_path(args.split_path, args.dataset_path)
    print(f"Split path: {split_path}")
    with open(split_path, "r", encoding="utf-8") as f:
        split_dict = json.load(f)
    dataset_path = resolve_dataset_path(args.dataset_path, split_dict.get(args.split, []))
    print(f"Dataset path: {dataset_path}")

    # 2. Check feature cache
    if args.feat_cache and os.path.isfile(args.feat_cache):
        print(f"Loading precomputed features from cache: {args.feat_cache}")
        cached = torch.load(args.feat_cache, map_location="cpu")
        features = cached["features"]
        labels = cached["labels"]
        target_classes = cached.get("classes", split_dict.get(args.split, []))
    else:
        # Load dataset & images
        img_paths, labels_list, target_classes = load_test_images(split_path, dataset_path, partition=args.split)
        transform = build_transform(size=224)
        ds = FruitTestDataset(img_paths, labels_list, transform)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        if args.zeroshot:
            print("Running Baseline: Zero-shot CLIP ViT-B/16 (without prompt learning)")
            # Load standard ViT-B/16 directly via clip module without requiring dassl
            url = clip._MODELS.get("ViT-B/16")
            model_path = clip._download(url) if url else None
            try:
                state_dict = torch.jit.load(model_path, map_location="cpu").state_dict() if model_path else None
            except RuntimeError:
                state_dict = torch.load(model_path, map_location="cpu") if model_path else None

            design_details = {
                "trainer": "MaPLe",
                "vision_depth": 0,
                "language_depth": 0,
                "vision_ctx": 0,
                "language_ctx": 0,
                "maple_length": 2
            }
            clip_model = clip.build_model(state_dict, design_details).to(device)
            features, labels = extract_features(clip_model, loader, device, is_maple=False)
            model_name = "Zero-shot CLIP (ViT-B/16)"
        else:
            print(f"Loading trained MaPLe model from: {args.model_dir}")
            # Build MaPLe architecture directly
            url = clip._MODELS.get("ViT-B/16")
            model_path = clip._download(url) if url else None
            try:
                state_dict = torch.jit.load(model_path, map_location="cpu").state_dict() if model_path else None
            except RuntimeError:
                state_dict = torch.load(model_path, map_location="cpu") if model_path else None

            design_details = {
                "trainer": "MaPLe",
                "vision_depth": 0,
                "language_depth": 0,
                "vision_ctx": 0,
                "language_ctx": 0,
                "maple_length": 2
            }
            clip_model = clip.build_model(state_dict, design_details)

            # Minimal CFG stand-in for CustomCLIP
            class _CfgObj:
                pass
            cfg = _CfgObj()
            cfg.INPUT = _CfgObj()
            cfg.INPUT.SIZE = (224, 224)
            cfg.TRAINER = _CfgObj()
            cfg.TRAINER.MAPLE = _CfgObj()
            cfg.TRAINER.MAPLE.N_CTX = 2
            cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"
            cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9
            cfg.TRAINER.MAPLE.PREC = "fp16"

            from trainers.maple import CustomCLIP
            classnames = [CLASS_NAME_MAP.get(c, c.replace("_", " ").replace("-", " ")) for c in target_classes]
            model = CustomCLIP(cfg, classnames, clip_model)

            # Find checkpoint
            ckpt_path = None
            if args.model_dir:
                candidates = [
                    os.path.join(args.model_dir, "MultiModalPromptLearner", "model-best.pth.tar"),
                    os.path.join(args.model_dir, "model-best.pth.tar"),
                    os.path.join(args.model_dir, "MultiModalPromptLearner", f"model.pth.tar-{args.load_epoch}"),
                    os.path.join(args.model_dir, f"model.pth.tar-{args.load_epoch}"),
                ]
                for cand in candidates:
                    if os.path.isfile(cand):
                        ckpt_path = cand
                        break
                if not ckpt_path:
                    found = glob.glob(os.path.join(args.model_dir, "**", "*.pth.tar*"), recursive=True)
                    if found:
                        ckpt_path = sorted(found)[-1]

            if ckpt_path and os.path.isfile(ckpt_path):
                print(f"Loading weights from: {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                state_dict = checkpoint.get("state_dict", checkpoint)
                if "prompt_learner.token_prefix" in state_dict:
                    del state_dict["prompt_learner.token_prefix"]
                if "prompt_learner.token_suffix" in state_dict:
                    del state_dict["prompt_learner.token_suffix"]
                model.load_state_dict(state_dict, strict=False)
            else:
                print("WARNING: Checkpoint not found, using initialized weights!")

            model.to(device)
            features, labels = extract_features(model, loader, device, is_maple=True)
            model_name = f"MaPLe (ViT-B/16, {args.model_dir})"

        if args.feat_cache:
            os.makedirs(os.path.dirname(os.path.abspath(args.feat_cache)), exist_ok=True)
            torch.save({"features": features, "labels": labels, "classes": target_classes}, args.feat_cache)
            print(f"Features saved to cache: {args.feat_cache}")

    # 3. Run Episodic Testing
    print("\n" + "=" * 60)
    print(f"  Running {args.n_episodes} Episodes ({args.n_way}-way {args.k_shot}-shot)")
    print(f"  Classes ({len(target_classes)}): {target_classes}")
    print("=" * 60)

    mean_acc, ci95, std_acc = run_episodic_evaluation(
        features=features,
        labels=labels,
        n_episodes=args.n_episodes,
        n_way=args.n_way,
        k_shot=args.k_shot,
        n_query=args.n_query,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print(f"  RESULT - Episodic {args.n_way}-way {args.k_shot}-shot Evaluation")
    print(f"  Split    : {args.split}")
    print(f"  Episodes : {args.n_episodes}")
    print("=" * 60)
    print(f"  Accuracy = {mean_acc:.2f} +/- {ci95:.2f}% (std: {std_acc:.2f}%)")
    print("=" * 60 + "\n")

    # 4. Save results to JSON
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        summary = {
            "model": "Zero-shot CLIP" if args.zeroshot else "MaPLe",
            "split": args.split,
            "n_way": args.n_way,
            "k_shot": args.k_shot,
            "n_query": args.n_query,
            "n_episodes": args.n_episodes,
            "mean_acc": round(mean_acc, 2),
            "ci95": round(ci95, 2),
            "std_acc": round(std_acc, 2),
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        print(f"Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
