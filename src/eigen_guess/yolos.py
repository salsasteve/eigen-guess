# """Main script to demonstrate low-rank approximation of a neural network."""
# import copy
# import json
# import os

# import matplotlib.pyplot as plt
# import requests
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from PIL import Image
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# from torch.utils.data import Dataset
# from transformers import AutoImageProcessor, AutoModelForObjectDetection

# from eigen_guess.low_rank import ModuleLowRank
# from torch.utils.data import DataLoader


# class COCODataset(Dataset):
#     def __init__(self, root_dir, annotation_file, transforms=None):
#         self.root_dir = root_dir
#         self.coco = COCO(annotation_file)
#         self.ids = sorted(self.coco.imgs.keys())
#         self.transforms = transforms

#     def __getitem__(self, index):
#         coco = self.coco
#         img_id = self.ids[index]
#         img_info = coco.loadImgs(img_id)[0]
#         path = img_info['file_name']
#         img = Image.open(os.path.join(self.root_dir, path)).convert('RGB')

#         if self.transforms is not None:
#             img = self.transforms(img)

#         ann_ids = coco.getAnnIds(imgIds=img_id)
#         anns = coco.loadAnns(ann_ids)
#         target = {'annotations': anns, 'image_id': img_id}
#         return img, target

#     def __len__(self):
#         return len(self.ids)


# def create_sample_model():
#     """Create a sample neural network with linear layers."""
#     model = nn.Sequential(
#         nn.Linear(64, 128),
#         nn.ReLU(),
#         nn.Linear(128, 256),
#         nn.ReLU(),
#         nn.Linear(256, 10)
#     )
#     return model

# def print_model_summary(model):
#     """Print the summary of the model."""
#     print("Model summary:")
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Linear):
#             print(f"{name}: {module}")

# def example():
#     """Example usage of the low-rank approximation."""
#     model = create_sample_model()

#     print("Original model:")
#     print_model_summary(model)

#     # Apply low-rank approximation
#     low_rank = ModuleLowRank()
#     model = low_rank(model)

#     print("\nModel after applying low-rank approximation:")
#     print_model_summary(model)


# def simple_test(model: nn.Module):
#     """Simple test model."""
#     url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#     image = Image.open(requests.get(url, stream=True).raw)
#     image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")

#     inputs = image_processor(images=image, return_tensors="pt")
#     outputs = model(**inputs)

#     # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
#     target_sizes = torch.tensor([image.size[::-1]])

#     results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
#         0
#     ]

#     print("results", results)

#     for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#         box = [round(i, 2) for i in box.tolist()]
#         print(
#             f"Detected {model.config.id2label[label.item()]} with confidence "
#             f"{round(score.item(), 3)} at location {box}"
#         )

# def compare_models_confidence(model1, model2, image_urls):
#     """Compare the average confidence score of two models over a set of images."""
#     image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")

#     def get_average_confidence(model):
#         total_confidence = 0.0
#         total_boxes = 0
#         for url in image_urls:
#             image = Image.open(requests.get(url, stream=True).raw)
#             inputs = image_processor(images=image, return_tensors="pt")
#             outputs = model(**inputs)

#             target_sizes = torch.tensor([image.size[::-1]])
#             results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

#             for score in results["scores"]:
#                 total_confidence += score.item()
#                 total_boxes += 1

#         if total_boxes == 0:
#             return 0
#         return total_confidence / total_boxes

#     avg_conf_model1 = get_average_confidence(model1)
#     avg_conf_model2 = get_average_confidence(model2)

#     print(f"Average confidence score for model 1 (original): {avg_conf_model1}")
#     print(f"Average confidence score for model 2 (low-rank): {avg_conf_model2}")

# image_urls = [
#     "http://images.cocodataset.org/val2017/000000039769.jpg",
#     # Add more image URLs for a broader comparison
# ]


# def cache_model_weights(model: nn.Module, name: str):
#     """Save the model weights to disk."""
#     torch.save(model.state_dict(), f"{name}_weights.pt")

# def load_cached_model_weights(name: str, model_class):
#     """Load the cached model weights into a new model instance."""
#     model = None
#     # Check if the model weights are cached
#     if os.path.exists(f"{name}_weights.pt"):
#         model = model_class()
#         model.load_state_dict(torch.load(f"{name}_weights.pt"))
#     return model

# def create_yolos_tiny_model(pretrained_model_name="hustvl/yolos-tiny"):
#     """Create a new instance of the YOLOS-tiny model."""
#     return AutoModelForObjectDetection.from_pretrained(pretrained_model_name)


# def visualize_detections(model, dataset, num_images=5, device='cpu'):
#     model.to(device)
#     model.eval()
#     image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
#     for idx in range(num_images):
#         img, target = dataset[idx]
#         img_id = target['image_id']
#         img = img.to(device)

#         with torch.no_grad():
#             inputs = image_processor(images=img, return_tensors="pt")
#             inputs = {k: v.to(device) for k, v in inputs.items()}
#             outputs = model(**inputs)

#         target_sizes = torch.tensor([img.size[::-1]]).to(device)
#         results = image_processor.post_process_object_detection(
#             outputs, threshold=0.5, target_sizes=target_sizes
#         )[0]

#         # Convert image tensor to NumPy array
#         img_np = img.cpu().numpy().transpose(1, 2, 0)

#         plt.figure(figsize=(12, 8))
#         plt.imshow(img_np)
#         ax = plt.gca()

#         for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#             box = box.detach().cpu().numpy()
#             xmin, ymin, xmax, ymax = box
#             width = xmax - xmin
#             height = ymax - ymin
#             rect = plt.Rectangle((xmin, ymin), width, height, fill=False, color='red', linewidth=2)
#             ax.add_patch(rect)
#             ax.text(xmin, ymin - 2, f"{model.config.id2label[label.item()]}: {score:.3f}",
#                     fontsize=12, color='red')
#         plt.axis('off')
#         plt.show()


# def evaluate_model(model, data_loader, device='cpu'):
#     model.to(device)
#     model.eval()
#     coco_gt = data_loader.dataset.coco
#     results = []
#     image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
#     for images, targets in data_loader:
#         images = images.to(device)

#         # Handle batch_size=1 case
#         if isinstance(targets, dict):
#             targets = [targets]

#         img_ids = [target['image_id'] for target in targets]

#         with torch.no_grad():
#             inputs = image_processor(images=images, return_tensors="pt", do_rescale=False)
#             inputs = {k: v.to(device) for k, v in inputs.items()}
#             outputs = model(**inputs)

#         # Post-process outputs
#         target_sizes = torch.tensor([[image.shape[-2], image.shape[-1]] for image in images]).to(device)
#         results_per_batch = image_processor.post_process_object_detection(
#             outputs, threshold=0.05, target_sizes=target_sizes
#         )

#         for img_id, result in zip(img_ids, results_per_batch):
#             boxes = result['boxes'].detach().cpu().numpy()
#             scores = result['scores'].detach().cpu().numpy()
#             labels = result['labels'].detach().cpu().numpy()

#             for box, score, label in zip(boxes, scores, labels):
#                 x_min, y_min, x_max, y_max = box
#                 width = x_max - x_min
#                 height = y_max - y_min
#                 detection = {
#                     "image_id": int(img_id),
#                     "category_id": int(label),
#                     "bbox": [x_min, y_min, width, height],
#                     "score": float(score),
#                 }
#                 results.append(detection)

#     # Save results to a JSON file
#     with open('results.json', 'w') as f:
#         json.dump(results, f)

#     # Load the results
#     coco_dt = coco_gt.loadRes('results.json')

#     # Initialize COCOeval object
#     coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

#     # Run evaluation
#     coco_eval.evaluate()
#     coco_eval.accumulate()
#     coco_eval.summarize()

#     # Return the mAP
#     mAP = coco_eval.stats[0]
#     print(f"Model mAP: {mAP:.4f}")
#     return mAP


# def main():
#     """Main function."""
#     model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")
#     model_org = copy.deepcopy(model)

#     print_model_summary(model_org)

#     low_rank = ModuleLowRank()
#     model_lr = low_rank(model)

#     print("\nModel after applying low-rank approximation:")
#     print_model_summary(model_lr)

#     # simple_test(model)
#     # simple_test(model_org)
# #     print("Visualizing Detections for Original Model:")

# #     transform = transforms.Compose([
# #     transforms.ToTensor(),
# #     ])

# #     train_dataset = COCODataset(
# #         root_dir='train2017/',
# #         annotation_file='annotations/instances_train2017.json',
# #         transforms=transform
# #     )

# #     val_dataset = COCODataset(
# #         root_dir='val2017/',
# #         annotation_file='annotations/instances_val2017.json',
# #         transforms=transform
# # )

# #     val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
# #     print("val_loader", val_loader)
# #     image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")

# #     # Evaluate the original model
# #     print("Evaluating Original Model:")
# #     mAP_org = evaluate_model(model_org, val_loader)

# #     # Evaluate the low-rank model
# #     print("\nEvaluating Low-Rank Model:")
# #     mAP_lr = evaluate_model(model_lr, val_loader)
# #     # visualize_detections(model_org, val_dataset)

#     # # Visualize detections for the low-rank model
#     # print("\nVisualizing Detections for Low-Rank Model:")
#     # visualize_detections(model_lr, val_dataset)


# main()
import torch
import time
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import copy
from eigen_guess.low_rank import ModuleLowRank


def run_inference(model, processor, image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs


def visualize_detections(image, outputs, threshold=0.5):
    boxes = outputs["pred_boxes"][0].detach().cpu()
    scores = outputs["pred_logits"][0].softmax(-1).max(-1)
    labels = scores.indices
    confidences = scores.values

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box, label, confidence in zip(boxes, labels, confidences):
        if (
            confidence > threshold and label != 91
        ):  # 91 is the background class in COCO
            xmin, ymin, xmax, ymax = box
            rect = plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                xmin,
                ymin - 5,
                f"{label.item()}:{confidence.item():.2f}",
                color="red",
            )
    plt.show()


def main():
    # Load the models
    model_org = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")
    model_org.eval()
    model_lr = ModuleLowRank()(copy.deepcopy(model_org))
    model_lr.eval()

    # Load the processor
    processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")

    # Load and preprocess the test image
    image_path = "path_to_test_image.jpg"  # Replace with your image path
    image = Image.open(image_path).convert("RGB")

    # Run inference with the original model
    start_time = time.time()
    outputs_org = run_inference(model_org, processor, image)
    time_org = time.time() - start_time

    # Run inference with the low-rank model
    start_time = time.time()
    outputs_lr = run_inference(model_lr, processor, image)
    time_lr = time.time() - start_time

    # Visualize detections
    print("Original Model Detections:")
    visualize_detections(image, outputs_org)

    print("Low-Rank Model Detections:")
    visualize_detections(image, outputs_lr)

    # Print inference times
    print(f"Original model inference time: {time_org:.4f} seconds")
    print(f"Low-rank model inference time: {time_lr:.4f} seconds")

    # If you have ground truth annotations, compute evaluation metrics here


if __name__ == "__main__":
    main()
