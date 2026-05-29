"""
train.py - Fine-tune YOLOv8 for retail shelf product detection
Optimized for Google Colab (free T4 GPU)
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch


def setup_sku110k(data_dir: str = "data/sku110k") -> dict:
    """
    Prepare SKU-110K dataset for YOLO training.
    
    SKU-110K: 11,762 shelf images, 1.7M product annotations
    Download: https://github.com/eg4000/SKU110K_CVPR19
    
    Expected structure:
        data/sku110k/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/
    """
    data_dir = Path(data_dir)
    
    # Create dataset YAML
    dataset_config = {
        'path': str(data_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {0: 'product'},  # Single class: generic product
        'nc': 1
    }
    
    yaml_path = data_dir / 'sku.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Dataset config saved to {yaml_path}")
    return dataset_config


def convert_sku110k_annotations(
    annotations_csv: str,
    output_dir: str,
    image_dir: str
):
    """
    Convert SKU-110K CSV annotations to YOLO format.
    
    SKU-110K format: x1,y1,x2,y2 (absolute pixels)
    YOLO format: class_id, x_center, y_center, width, height (normalized 0-1)
    """
    import csv
    from PIL import Image
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read annotations grouped by image
    annotations = {}
    with open(annotations_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row['image_name']
            if img_name not in annotations:
                annotations[img_name] = []
            
            annotations[img_name].append({
                'x1': float(row['x1']),
                'y1': float(row['y1']),
                'x2': float(row['x2']),
                'y2': float(row['y2'])
            })
    
    # Convert to YOLO format
    converted = 0
    for img_name, boxes in annotations.items():
        img_path = Path(image_dir) / img_name
        if not img_path.exists():
            continue
        
        # Get image dimensions
        with Image.open(img_path) as img:
            img_w, img_h = img.size
        
        # Write label file
        label_name = Path(img_name).stem + '.txt'
        label_path = output_dir / label_name
        
        with open(label_path, 'w') as f:
            for box in boxes:
                # Convert to YOLO format
                x_center = ((box['x1'] + box['x2']) / 2) / img_w
                y_center = ((box['y1'] + box['y2']) / 2) / img_h
                width = (box['x2'] - box['x1']) / img_w
                height = (box['y2'] - box['y1']) / img_h
                
                # Clamp to [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        converted += 1
    
    print(f"Converted {converted} images to YOLO format")
    return converted


def train_model(
    data_yaml: str = "data/sku110k/sku.yaml",
    model_size: str = "n",  # n, s, m, l, x
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project: str = "runs/sku_training",
    name: str = "sku_yolov8"
):
    """
    Train YOLOv8 on SKU-110K dataset.
    
    Args:
        data_yaml: Path to dataset YAML
        model_size: Model size (n=Nano, s=Small, m=Medium, l=Large, x=XLarge)
        epochs: Training epochs (50-100 recommended)
        imgsz: Input image size
        batch: Batch size (adjust based on GPU memory)
        device: GPU device ID or 'cpu'
        project: Output directory
        name: Run name
    """
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected. Training will be very slow.")
        device = 'cpu'
        batch = 4  # Reduce batch for CPU
    
    # Load pretrained model
    model = YOLO(f"yolov8{model_size}.pt")
    
    # Training arguments optimized for shelf detection
    args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'project': project,
        'name': name,
        'patience': 10,  # Early stopping
        'save': True,
        'save_period': 10,
        'cache': True,  # Cache images in RAM for speed
        'workers': 8,
        
        # Augmentation for shelf scenarios
        'hsv_h': 0.015,  # Hue
        'hsv_s': 0.7,    # Saturation
        'hsv_v': 0.4,    # Value
        'degrees': 5.0,  # Rotation
        'translate': 0.1,
        'scale': 0.5,
        'shear': 2.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,   # Mosaic augmentation
        'mixup': 0.1,
        'copy_paste': 0.1,
        
        # Loss weights
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # Optimizer
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        
        # Logging
        'verbose': True
    }
    
    print(f"\n{'='*50}")
    print(f"Starting Training: YOLOv8{model_size.upper()} on SKU-110K")
    print(f"Epochs: {epochs}, Image Size: {imgsz}, Batch: {batch}")
    print(f"{'='*50}\n")
    
    # Train
    results = model.train(**args)
    
    # Validate
    print("\nValidating best model...")
    metrics = model.val()
    
    print(f"\n{'='*50}")
    print("Training Complete!")
    print(f"Best mAP50: {metrics.box.map50:.4f}")
    print(f"Best mAP50-95: {metrics.box.map:.4f}")
    print(f"Model saved to: {project}/{name}/weights/best.pt")
    print(f"{'='*50}")
    
    return model, results


def export_model(
    model_path: str = "runs/sku_training/sku_yolov8/weights/best.pt",
    format: str = "onnx",
    imgsz: int = 640
):
    """
    Export trained model to deployment format.
    
    Formats: onnx, torchscript, openvino, tensorrt, coreml
    """
    model = YOLO(model_path)
    
    print(f"Exporting to {format.upper()}...")
    model.export(format=format, imgsz=imgsz, half=True)
    
    print(f"Export complete: {model_path.replace('.pt', f'.{format}')}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv8 for shelf monitoring")
    parser.add_argument("--setup", action="store_true", help="Setup dataset")
    parser.add_argument("--convert", action="store_true", help="Convert annotations")
    parser.add_argument("--train", action="store_true", help="Start training")
    parser.add_argument("--export", action="store_true", help="Export model")
    parser.add_argument("--data-dir", default="data/sku110k", help="Dataset directory")
    parser.add_argument("--model", default="n", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    
    args = parser.parse_args()
    
    if args.setup:
        setup_sku110k(args.data_dir)
    
    if args.convert:
        # Example conversion paths
        convert_sku110k_annotations(
            annotations_csv=f"{args.data_dir}/annotations.csv",
            output_dir=f"{args.data_dir}/labels/train",
            image_dir=f"{args.data_dir}/images/train"
        )
    
    if args.train:
        model, results = train_model(
            data_yaml=f"{args.data_dir}/sku.yaml",
            model_size=args.model,
            epochs=args.epochs,
            batch=args.batch
        )
    
    if args.export:
        export_model()
    
    # Default: run all
    if not any([args.setup, args.convert, args.train, args.export]):
        print("Usage: python train.py --train")
        print("       python train.py --setup --convert --train --export")
