from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    
    # Optimal fabric defect detection training configuration
    model.train(
        data="dataset/data.yaml",
        epochs=100,  # More epochs for fabric patterns
        imgsz=640,
        batch=16,
        project="runs",
        name="fabric-defect-yolov8n",
        device=0,  # Use GPU 0
        workers=8,  # Parallel data loading
        patience=50,  # Early stopping patience
        pretrained=True,
        optimizer='auto',
        
        # === CRITICAL AUGMENTATIONS FOR FABRIC ===
        # Color variations (simulate dye lots, lighting)
        hsv_h=0.015,   # Hue variation
        hsv_s=0.7,     # Saturation variation (important for prints)
        hsv_v=0.4,     # Value/brightness variation
        
        # Geometric (conservative for fabrics)
        degrees=0.0,   # NO rotation - camera is fixed
        translate=0.1, # Small translation
        scale=0.5,     # Scale variation
        shear=2.0,     # Small shear (fabric stretch)
        
        # Flips (fabric can run both directions)
        fliplr=0.5,    # 50% horizontal flip
        flipud=0.0,    # NO vertical flip (defects have orientation)
        
        # Advanced augmentations
        mosaic=1.0,    # 100% mosaic - excellent for small defects
        mixup=0.2,     # 20% mixup
        copy_paste=0.3, # 30% copy-paste (BEST for fabric)
        
        # Image quality variations
        perspective=0.0001,  # Minimal perspective
        brightness=0.2,      # Lighting changes
        contrast=0.3,        # Contrast variations
        saturation=0.5,      # Important for colored fabrics
        
        # === OPTIMIZATION SETTINGS ===
        lr0=0.01,      # Initial learning rate
        lrf=0.01,      # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,   # Cosine LR scheduler
        box=7.5,       # Box loss gain
        cls=0.5,       # Class loss gain (lower for few classes)
        dfl=1.5,       # Distribution Focal Loss
        
        # === TRAINING CONFIG ===
        label_smoothing=0.0,  # Keep 0 for precise defect boundaries
        nbs=64,        # Nominal batch size
        overlap_mask=True,  # For future segmentation
        single_cls=False,   # Multi-class detection
        amp=True,      # Mixed precision training
        fraction=1.0,  # Use all data
        seed=42,       # Reproducibility
        deterministic=True,
        val=True,      # Validate during training
        plots=True,    # Generate plots
        save=True,     # Save checkpoints
        save_period=10,# Save every 10 epochs
        resume=False,  # Fresh training
        cache='ram',   # Cache in RAM for speed
        verbose=True,  # Show progress
        exist_ok=True, # Overwrite existing
    )

if __name__ == "__main__":
    main()