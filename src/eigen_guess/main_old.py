"""Main script to demonstrate low-rank approximation of a neural network."""

import copy
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from eigen_guess.low_rank import ModuleLowRank


def create_sample_model():
    """Create a sample neural network with linear layers."""
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    return model


def print_model_summary(model):
    """Print the summary of the model."""
    print("Model summary:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"{name}: {module}")


def run_example():
    """Example usage of the low-rank approximation."""
    model = create_sample_model()

    print("Original model:")
    print_model_summary(model)

    # Apply low-rank approximation
    low_rank = ModuleLowRank()
    model = low_rank(model)

    print("\nModel after applying low-rank approximation:")
    print_model_summary(model)


def get_yolos_model():
    try:
        # Try to load the model from local files only
        model = AutoModelForObjectDetection.from_pretrained(
            "hustvl/yolos-tiny", local_files_only=True
        )
    except OSError:
        # If not found locally, download the model from the internet
        model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")
    model_org = copy.deepcopy(model)
    return model, model_org


def print_model_summary(model):
    """Print the summary of the model."""
    print("Model summary:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"{name}: {module}")


def get_low_rank_model(model):
    """Get the low-rank approximated model, loading from cache if available."""
    low_rank_model_path = "model_lr.pth"

    if os.path.exists(low_rank_model_path):
        # Load low-rank model from cache
        print("Loading low-rank model from cache...")
        model_lr = torch.load(low_rank_model_path)
    else:
        # Apply low-rank approximation
        low_rank = ModuleLowRank()
        model_lr = low_rank(model)

        # Save the low-rank model
        torch.save(model_lr, low_rank_model_path)
        print("Low-rank model saved to cache.")

    return model_lr


def get_low_rank_model(model, num_layers=-1):
    """Get the low-rank approximated model, loading from cache if available."""
    low_rank_model_path = f"model_lr{num_layers}.pth"

    if os.path.exists(low_rank_model_path):
        # Load low-rank model from cache
        print("Loading low-rank model from cache...")
        model_lr = torch.load(low_rank_model_path)
    else:
        # Apply low-rank approximation
        low_rank = ModuleLowRank(num_layers=num_layers)
        model_lr = low_rank(model)

        # Save the low-rank model
        torch.save(model_lr, low_rank_model_path)
        print("Low-rank model saved to cache.")

    return model_lr


class COCODataset(Dataset):
    def __init__(self, root_dir, annotation_file, transforms=None):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.ids = sorted(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_info = coco.loadImgs(img_id)[0]
        path = img_info["file_name"]
        img = Image.open(os.path.join(self.root_dir, path)).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        target = {"annotations": anns, "image_id": img_id}
        return img, target

    def __len__(self):
        return len(self.ids)


def get_data(fraction=1.0):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset = COCODataset(
        root_dir="train2017/",
        annotation_file="annotations/instances_train2017.json",
        transforms=transform,
    )

    val_dataset = COCODataset(
        root_dir="val2017/",
        annotation_file="annotations/instances_val2017.json",
        transforms=transform,
    )

    if fraction < 1.0:
        # Set random seed for reproducibility
        random_seed = 42
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Reduce train_dataset
        train_size = len(train_dataset)
        train_subset_size = max(1, int(train_size * fraction))
        train_indices = torch.randperm(train_size)[:train_subset_size]
        train_dataset = Subset(train_dataset, train_indices)

        # Reduce val_dataset
        val_size = len(val_dataset)
        val_subset_size = max(1, int(val_size * fraction))
        val_indices = torch.randperm(val_size)[:val_subset_size]
        val_dataset = Subset(val_dataset, val_indices)

    return train_dataset, val_dataset


def get_original_dataset(dataset):
    while isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    return dataset


def evaluate_model(model, data_loader, device="cpu"):
    model.to(device)
    model.eval()

    # Retrieve the original dataset
    original_dataset = get_original_dataset(data_loader.dataset)
    coco_gt = original_dataset.coco

    results = []
    image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        img_ids = [target["image_id"] for target in targets]

        with torch.no_grad():
            inputs = image_processor(
                images=images, return_tensors="pt", do_rescale=False
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

        # Post-process outputs
        target_sizes = torch.tensor(
            [[img.shape[-2], img.shape[-1]] for img in images]
        ).to(device)
        results_per_batch = image_processor.post_process_object_detection(
            outputs, threshold=0.05, target_sizes=target_sizes
        )

        for img_id, result in zip(img_ids, results_per_batch):
            boxes = result["boxes"].detach().cpu().numpy()
            scores = result["scores"].detach().cpu().numpy()
            labels = result["labels"].detach().cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                x_min, y_min, x_max, y_max = box
                width = x_max - x_min
                height = y_max - y_min
                detection = {
                    "image_id": int(img_id),
                    "category_id": int(label),
                    "bbox": [
                        float(x_min),
                        float(y_min),
                        float(width),
                        float(height),
                    ],
                    "score": float(score),
                }
                results.append(detection)

    # Save results to a JSON file
    with open("results.json", "w") as f:
        json.dump(results, f)

    # Load the results
    coco_dt = coco_gt.loadRes("results.json")

    # Initialize COCOeval object
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Return the mAP
    mAP = coco_eval.stats[0]
    print(f"Model mAP: {mAP:.4f}")
    return mAP


def collate_fn(batch):
    images, targets = zip(*batch)
    return images, targets


def main():
    """Main function."""
    model, model_org = get_yolos_model()

    print_model_summary(model_org)

    # Apply low-rank approximation
    model_lr = get_low_rank_model(model, num_layers=5)

    print("\nModel after applying low-rank approximation:")
    print_model_summary(model_lr)

    train_dataset, val_dataset = get_data(fraction=0.1)

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    print("Evaluating Original Model:")
    mAP_org = evaluate_model(model_org, val_loader)

    # Evaluate the low-rank model
    print("\nEvaluating Low-Rank Model:")
    mAP_lr = evaluate_model(model_lr, val_loader)
