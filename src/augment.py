import cv2
import albumentations as A
from pathlib import Path                
import shutil


# --- transformation pipeline ---
transformer = A.Compose([
    # --- flip & rotate ---
    A.Resize(width=640, height=640),            # Resize to a fixed size (640x640)
    A.HorizontalFlip(p=0.5),                    # Horizontal flip with 50% probability
    A.VerticalFlip(p=0.2),                      # Vertical flip with 20% probability
    A.RandomRotate90(p=0.4),                    # Random 90-degree rotation with 40% probability
    A.Rotate(limit=30, p=0.4, border_mode=4),   # Random rotation up to 30 degrees with black borders

    # --- crop & pad ---
    A.RandomCrop(width=500, height=500, p=0.8), # Crop to 500x500 with 80% probability

    # --- color & noise ---
    A.RandomRain(
        rain_type='heavy',  
        drop_length=20, 
        drop_width=1, 
        p=0.3
    ),                                          # Add heavy rain with 30% probability
    A.RandomBrightnessContrast(
        brightness_limit=0.3, 
        contrast_limit=0.3, 
        p=0.6
    ),                                          # Adjust brightness and contrast with 60% probability
    A.HueSaturationValue(
        hue_shift_limit=20, 
        sat_shift_limit=30, 
        val_shift_limit=20, 
        p=0.4
    )                                           # Manipulate hue, saturation, and value with 40% probability

], bbox_params=A.BboxParams(
    format='yolo',                             # Use YOLO format for bounding boxes
    label_fields=['class_labels'],             # Specify the field containing class labels
    min_visibility=0.3,                        # Delete bounding boxes that are less than 30% visible
    min_area=10                                # Delete bounding boxes with an area smaller than 10 pixels
))

# --- function to apply augmentation to an image and its bounding boxes ---
def augment_image(image_path, bboxes, class_labels):
    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply transformations
    augmented = transformer(image=image, bboxes=bboxes, class_labels=class_labels)
    
    return augmented['image'], augmented['bboxes'], augmented['class_labels']

# --- function to save augmented image and labels ---
def save_augmented_image(augmented_image, augmented_bboxes, augmented_class_labels, output_image_path, output_label_path):
    # Save augmented image
    augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_image_path), augmented_image_bgr)

    # Save augmented bounding boxes and class labels in YOLO format
    with open(output_label_path, 'w') as f:
        for bbox, class_label in zip(augmented_bboxes, augmented_class_labels):
            bbox_str = ' '.join(map(str, bbox))
            f.write(f"{class_label} {bbox_str}\n")


# --- main execution ---
if __name__ == '__main__':
    # apply augmentation to all images in the training set and save the augmented images and labels only consider classes 0 and 4
    labels_dir = Path("data/raw/train/labels/")
    label_files = list(labels_dir.glob("*.txt"))
    for label_file in label_files:
        # Load bounding boxes and class labels from the label file
        with open(label_file, 'r') as f:
            lines = f.readlines()
            bboxes = []
            class_labels = []
            chick_classes = []
            for line in lines:
                parts = line.strip().split()
                chick_classes.append(int(parts[0]))
            if 0 in chick_classes or 4 in chick_classes:
                for line in lines:
                    parts = line.strip().split()
                    class_label = int(parts[0])
                    bbox = list(map(float, parts[1:]))
                    bboxes.append(bbox)
                    class_labels.append(class_label)

                # apply augmentation and save augmented image and labels only if they don't already exist
                image_path = label_file.parent.parent / "images" / label_file.name.replace(".txt", ".jpg")
                output_image_path = Path("data/augmented/images/") / label_file.name.replace(".txt", "_aug.jpg")
                output_label_path = Path("data/augmented/labels/") / label_file.name.replace(".txt", "_aug.txt")
                if output_image_path.exists() and output_label_path.exists():
                    continue  
                augmented_image, augmented_bboxes, augmented_class_labels = augment_image(image_path, bboxes, class_labels)

                # Save augmented image and labels
                save_augmented_image(augmented_image, augmented_bboxes, augmented_class_labels, output_image_path, output_label_path)
                
# --- copy augmented images and labels to the training dataset ---
    original_images_dir = Path("data/raw/train/images/")
    original_labels_dir = Path("data/raw/train/labels/")
    augmented_images_dir = Path("data/augmented/images/")
    augmented_labels_dir = Path("data/augmented/labels/")

    for image_file in augmented_images_dir.glob("*.jpg"):
        shutil.copy(image_file, original_images_dir / image_file.name)

    for label_file in augmented_labels_dir.glob("*.txt"):
        shutil.copy(label_file, original_labels_dir / label_file.name)


