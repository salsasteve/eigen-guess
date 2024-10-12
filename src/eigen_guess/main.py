"""evaluation script for Yolos-tiny model and low-rank approximation model."""
import argparse
import copy
import time

import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import YolosForObjectDetection, YolosImageProcessor

from eigen_guess.low_rank import ModuleLowRank


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate models on COCO validation set"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=100,
        help="Number of images to process",
    )
    return parser.parse_args()


def load_coco_dataset(ann_file):
    """Load COCO dataset using the provided annotation file."""
    return COCO(ann_file)


def get_image_ids(coco, num_images):
    """Get image IDs from the COCO dataset."""
    img_ids = coco.getImgIds()
    if num_images > 0:
        img_ids = img_ids[:num_images]
    return img_ids


def load_image(coco, image_id, img_dir):
    """Load an image from the COCO dataset."""
    img_info = coco.loadImgs(image_id)[0]
    img_path = f"{img_dir}/{img_info['file_name']}"
    # Load image and convert to RGB
    image = Image.open(img_path).convert("RGB")
    return image, img_info


def process_outputs(
    outputs, image_id, coco, processor, model_config, coco_name2id
):
    """Process model outputs to get detections."""
    # Get the original image size
    image_info = coco.loadImgs(image_id)[0]
    orig_size = torch.tensor([[image_info["height"], image_info["width"]]]).to(
        outputs.logits.device
    )

    # Post-process outputs
    results = processor.post_process_object_detection(
        outputs, threshold=0.05, target_sizes=orig_size
    )[0]

    detections = []
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        score = score.item()
        label = label.item()
        box = box.tolist()

        if label in model_config.id2label:
            category_name = model_config.id2label[label]
            category_id = coco_name2id.get(category_name, None)
            if category_id is None:
                continue  # Skip if category not in COCO categories
        else:
            continue  # Skip unknown labels

        # Box format is [x_min, y_min, x_max, y_max]; convert to [x_min, y_min, width, height]
        x_min, y_min, x_max, y_max = box
        width_box = x_max - x_min
        height_box = y_max - y_min
        bbox = [x_min, y_min, width_box, height_box]

        detection = {
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "score": score,
        }
        detections.append(detection)
    return detections


def run_inference(
    model, processor, coco, img_ids, img_dir, model_config, coco_name2id
):
    results = []
    device = next(model.parameters()).device
    model.eval()
    for idx, img_id in enumerate(img_ids):
        image, _ = load_image(coco, img_id, img_dir)
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        res = process_outputs(
            outputs, img_id, coco, processor, model_config, coco_name2id
        )
        results.extend(res)
        # num_detections = len(res)
        # print(f"Image ID {img_id}: {num_detections} detections")
        if (idx + 1) % 100 == 0 or idx == len(img_ids) - 1:
            print(f"Processed {idx + 1}/{len(img_ids)} images")
    return results


def evaluate_results(coco_gt, results):
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats


def main():
    args = parse_args()
    num_images = args.num_images

    # Paths to COCO dataset
    ann_file = r"Q:\research\eigen-guess\coco\annotations\instances_val2017.json"  # Adjust path
    img_dir = r"Q:\research\eigen-guess\coco\val2017"  # Adjust path

    # Load COCO dataset
    coco = load_coco_dataset(ann_file)
    img_ids = get_image_ids(coco, num_images)

    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_org = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny").to(
        device
    )
    model_lr = ModuleLowRank()(copy.deepcopy(model_org)).to(device)

    # Load processor
    processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

    # Create category mapping
    model_config = model_org.config
    coco_categories = coco.loadCats(coco.getCatIds())
    coco_name2id = {cat["name"]: cat["id"] for cat in coco_categories}

    # Run inference with the original model
    print("Running inference with the original model...")
    start_time = time.time()
    results_org = run_inference(
        model_org, processor, coco, img_ids, img_dir, model_config, coco_name2id
    )
    time_org = time.time() - start_time
    print(f"Original model inference time: {time_org:.2f}s")

    # Run inference with the low-rank model
    print("\nRunning inference with the low-rank model...")
    start_time = time.time()
    results_lr = run_inference(
        model_lr, processor, coco, img_ids, img_dir, model_config, coco_name2id
    )
    time_lr = time.time() - start_time
    print(f"Low-rank model inference time: {time_lr:.2f}s")

    # Evaluate original model results
    print("\nEvaluating the original model...")
    stats_org = evaluate_results(coco, results_org)

    # Evaluate low-rank model results
    print("\nEvaluating the low-rank model...")
    stats_lr = evaluate_results(coco, results_lr)

    # Compare metrics
    print("\nComparison of evaluation metrics:")
    metrics = [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR1",
        "AR10",
        "AR100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    for idx, metric in enumerate(metrics):
        print(
            f"{metric}: Original = {stats_org[idx]:.3f}, Low-Rank = {stats_lr[idx]:.3f}"
        )


if __name__ == "__main__":
    main()
