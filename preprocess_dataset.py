import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import shutil
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_ROOT = 'FaceShape_Dataset'
OUTPUT_ROOT = 'Preprocessed_data'
IMAGE_SIZE = (224, 224)
PADDING_RATIO = 0.2 

# --- Initialize MediaPipe Face Detection with the new Tasks API ---
MODEL_PATH = 'models/blaze_face_short_range.tflite'

# Check if the model file exists, if not, provide a warning.
if not os.path.exists(MODEL_PATH):
    print(f"Warning: Face detection model '{MODEL_PATH}' not found.")
    print("Please download 'blaze_face_short_range.tflite' from MediaPipe's GitHub or documentation")
    

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.7)
face_detector = vision.FaceDetector.create_from_options(options)

# --- Function to process a single image ---
def process_image(image_path, output_path, image_size, padding_ratio):
    img = cv2.imread(image_path)
    if img is None:
        # print(f"Warning: Could not read image {image_path}. Skipping.")
        return False

    # Convert the image to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    detection_result = face_detector.detect(mp_image)

    if detection_result.detections:
        detection = detection_result.detections[0]
        bbox_mp = detection.bounding_box 

        ih, iw, _ = img.shape
        x, y, w, h = bbox_mp.origin_x, bbox_mp.origin_y, bbox_mp.width, bbox_mp.height

        # Calculate padding
        pad_x = int(w * padding_ratio)
        pad_y = int(h * padding_ratio)

        # Apply padding and ensure coordinates are within image bounds
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(iw, x + w + pad_x)
        y2 = min(ih, y + h + pad_y)

        cropped_face = img[y1:y2, x1:x2]

        if cropped_face.shape[0] > 0 and cropped_face.shape[1] > 0:
            resized_face = cv2.resize(cropped_face, image_size)
            cv2.imwrite(output_path, resized_face)
            return True
        else:
            print(f"Warning: Cropped face is empty for {image_path}. Skipping.")
            return False
    else:
        print(f"No face detected in {image_path}. Skipping.")
        return False

# --- Main preprocessing loop ---
def preprocess_dataset(dataset_root, output_root, image_size, padding_ratio):
    for split_type in ['training_set', 'testing_set']:
        input_split_dir = os.path.join(dataset_root, split_type)
        output_split_dir = os.path.join(output_root, split_type)

        if not os.path.exists(input_split_dir):
            print(f"Error: {input_split_dir} not found. Please check DATASET_ROOT.")
            return

        print(f"Processing {split_type}...")
        for category in os.listdir(input_split_dir):
            input_category_dir = os.path.join(input_split_dir, category)
            output_category_dir = os.path.join(output_split_dir, category)

            if not os.path.isdir(input_category_dir):
                continue

            if not os.path.exists(output_category_dir):
                os.makedirs(output_category_dir)

            image_files = [f for f in os.listdir(input_category_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            processed_count = 0
            with tqdm(total=len(image_files), desc=f"  {category}") as pbar:
                for filename in image_files:
                    image_path = os.path.join(input_category_dir, filename)
                    output_path = os.path.join(output_category_dir, filename)

                    if process_image(image_path, output_path, image_size, padding_ratio):
                        processed_count += 1
                    pbar.update(1)
            print(f"    Processed {processed_count}/{len(image_files)} images for {category}.")

    face_detector.close() # Release MediaPipe resources
    print("\nPreprocessing complete!")
    print(f"Preprocessed dataset saved to: {output_root}")

if __name__ == "__main__":
    # Ensure the output directory is clean before starting
    if os.path.exists(OUTPUT_ROOT):
        print(f"Deleting existing {OUTPUT_ROOT} directory...")
        shutil.rmtree(OUTPUT_ROOT)
    os.makedirs(OUTPUT_ROOT)

    preprocess_dataset(DATASET_ROOT, OUTPUT_ROOT, IMAGE_SIZE, PADDING_RATIO)