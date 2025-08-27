import cv2
import mediapipe as mp
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf

# CONFIGURATION
MODEL_PATH_CLASSIFIER = 'models/face_shape_classifier_5_classes.h5'
MODEL_PATH_FACE_DETECTOR = 'models/blaze_face_short_range.tflite'
PADDING_RATIO = 0.2
FACE_SHAPE_LABELS = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']


def run_face_detection():
    """
    Enhanced real-time face detection with user-friendly experience:
    - Shows bounding box with margin
    - Displays instructions on screen
    - Adds countdown before capturing face
    - Crops + saves face once as 'face.png'
    """
    # Initialize MediaPipe's Face Detection solution
    mp_face_detection = mp.solutions.face_detection

    FACE_MARGIN = 25
    CAPTURE_DELAY = 5
    FACE_SIZE = (224, 224)
    
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:
        # Webcam setup
        cap = cv2.VideoCapture(0)

        # Check if the webcam is opened correctly
        if not cap.isOpened():
            print("Error: Could not open the webcam.")
            return

        print("Press 'q' to exit the application.")
        print("A cropped face will be saved as 'face.png' after countdown.")
        
        captured = False
        countdown_start = None


        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Error: Could not read frame.")
                break

            # Flip the frame horizontally for a selfie-view display
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB for MediaPipe processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # To improve performance, optionally mark the frame as not writeable
            frame_rgb.flags.writeable = False
            results = face_detection.process(frame_rgb)
            frame_rgb.flags.writeable = True

            # Draw the detection if found
            if results.detections:
                for detection in results.detections:
                    # Get the bounding box from the detection object
                    bbox_data = detection.location_data.relative_bounding_box
                    
                    img_height, img_width, _ = frame.shape

                    # Calculate the absolute coordinates of the bounding box
                    x_min = int(bbox_data.xmin * img_width) - FACE_MARGIN
                    y_min = int(bbox_data.ymin * img_height) - FACE_MARGIN
                    width = int(bbox_data.width * img_width) + 2*FACE_MARGIN
                    height = int(bbox_data.height * img_height) + 2*FACE_MARGIN

                    # Ensure coordinates are within image boundaries
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(img_width, x_min + width), min(img_height, y_min + height)

                    # Draw L-shaped corners for a clean aesthetic
                    CORNER_COLOR = (255, 0, 0)
                    CORNER_THICKNESS = 2
                    CORNER_LENGTH = 25
                    
                    # Top-left corner
                    cv2.line(frame, (x_min, y_min), (x_min + CORNER_LENGTH, y_min), CORNER_COLOR, CORNER_THICKNESS)
                    cv2.line(frame, (x_min, y_min), (x_min, y_min + CORNER_LENGTH), CORNER_COLOR, CORNER_THICKNESS)
                    
                    # Top-right corner
                    cv2.line(frame, (x_max, y_min), (x_max - CORNER_LENGTH, y_min), CORNER_COLOR, CORNER_THICKNESS)
                    cv2.line(frame, (x_max, y_min), (x_max, y_min + CORNER_LENGTH), CORNER_COLOR, CORNER_THICKNESS)

                    # Bottom-left corner
                    cv2.line(frame, (x_min, y_max), (x_min + CORNER_LENGTH, y_max), CORNER_COLOR, CORNER_THICKNESS)
                    cv2.line(frame, (x_min, y_max), (x_min, y_max - CORNER_LENGTH), CORNER_COLOR, CORNER_THICKNESS)

                    # Bottom-right corner
                    cv2.line(frame, (x_max, y_max), (x_max - CORNER_LENGTH, y_max), CORNER_COLOR, CORNER_THICKNESS)
                    cv2.line(frame, (x_max, y_max), (x_max, y_max - CORNER_LENGTH), CORNER_COLOR, CORNER_THICKNESS)

                    if not captured:
                        if countdown_start is None:
                            countdown_start = time.time()

                        elapsed = time.time()-countdown_start
                        remaining = int(CAPTURE_DELAY-elapsed)

                        if remaining>0:
                            cv2.putText(frame, f"Hold still... capturing in {remaining}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        else:
                            # Capture face once
                            cropped_face = frame[y_min:y_max, x_min:x_max]
                            if cropped_face.size > 0:
                                cropped_face = cv2.resize(cropped_face, FACE_SIZE)
                                cv2.imwrite("face.png", cropped_face)
                                print("Face saved as face.png")
                                captured = True
                    else:
                        cv2.putText(frame, "Face captured successfully!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

            else:
                countdown_start = None
                if not captured:
                    cv2.putText(frame, "Align your face in the box...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)    
            
            # Display the main processed frame
            cv2.imshow("Real-time Face Detection", frame)

            # Exit loop on pressing 'q'
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def show_face_shape_results(cropped_face_path, predictions, top_k=4):
    """
    Display the cropped face with an aesthetic result card and styled bar chart.
    """
    # Read cropped face
    face_img = cv2.imread(cropped_face_path)
    if face_img is None:
        print("Error: Could not read cropped face image.")
        return

    # Convert BGR to RGB for matplotlib
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    # Sort predictions by probability (descending) & select top_k
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_k]
    shapes = [s for s, _ in sorted_preds]
    probs = [p * 100 for _, p in sorted_preds]  

    # Assign colors for consistency
    colors = {
        "Oval": "#4A90E2",     # Blue
        "Square": "#50E3C2",   # Teal
        "Round": "#F5A623",    # Orange
        "Heart": "#E94E77",    # Pink
        "Oblong": "#7ED321"    # Green
    }
    bar_colors = [colors.get(s, "#999999") for s in shapes]

    # Create figure with dark background
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6))

    # Show cropped face with title
    ax1.imshow(face_img)
    ax1.axis("off")
    top_shape = shapes[0]
    ax1.set_title(f"Detected Shape: {top_shape.upper()}",
                  fontsize=14, fontweight="bold", color="#FFD700")  

    # Initialize horizontal bar chart with zeros
    bars = ax2.barh(shapes, [0]*len(probs), color=bar_colors, alpha=0.9)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Confidence (%)", fontsize=12, color="white")
    ax2.set_title("Top Face Shape Predictions", fontsize=13, fontweight="bold")
    ax2.tick_params(axis="y", colors="white", labelsize=11)
    ax2.tick_params(axis="x", colors="white")
    fig.patch.set_facecolor("#1E1E1E")  # dark bg

    # Add empty annotations for updating during animation
    annotations = []
    for i, _ in enumerate(probs):
        ann = ax2.text(0, i, "", va="center", fontsize=11, color="white")
        annotations.append(ann)

    # Animation update function
    def update(frame):
        progress = frame / 100  # normalize [0,1]
        for bar, target, ann, shape in zip(bars, probs, annotations, shapes):
            value = target * progress
            bar.set_width(value)
            ann.set_text(f"{value:.1f}%")
            ann.set_x(value + 2)
        return list(bars) + annotations
    
    # Run animation
    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 101, 2),
                                  interval=40, blit=False, repeat=False)

    plt.tight_layout()
    plt.show()

def analyze_face_shape(model, img_path):
    """
    Analyzes the face shape using the trained classifier.
    Args:
        model: The loaded Keras model.
        img_path: Preprocessed face image (numpy array, ready for prediction).
    Returns:
        confidence_scores: dict, probabilities for all classes.
    """

    img_array = cv2.imread(img_path)

    if img_array is None:
        print(f"Error: Could not read image from path {img_path}")
        return None
    
    # Preprocess the image for the model
    normalized_img = img_array / 255.0
    input_tensor = np.expand_dims(normalized_img, axis=0)

    # Run prediction
    predictions = model.predict(input_tensor, verbose=0)[0]
    
    # Build dictionary of all confidence scores
    confidence_scores = {
        label: float(predictions[i])
        for i, label in enumerate(FACE_SHAPE_LABELS)
    }

    return confidence_scores


if __name__ == "__main__":
    # --- Acknowledged Limitation for this demo ---
    print("Note: The model was trained primarily on female faces. "
      "Accuracy for male faces may be lower due to a gender-imbalanced dataset.")
    
    # Load classifier model
    print("Loading classifier model...")
    try:
        face_shape_model = tf.keras.models.load_model(MODEL_PATH_CLASSIFIER)
        print("Face shape classifier loaded successfully.")
    except Exception as e:
        print(f"Error loading face shape classifier: {e}")
        face_shape_model = None

    if face_shape_model:
        print("Starting face detection and capture...")
        run_face_detection()

        cropped_face_path = "face.png"
        if not os.path.exists(cropped_face_path):
            print("No face was captured. Exiting.")
        else:
            print("\nFace captured. Proceeding with analysis.")

            confidence_scores = analyze_face_shape(face_shape_model, cropped_face_path)

            if confidence_scores:
                print("Analysis complete. Displaying results.")
                show_face_shape_results(cropped_face_path, confidence_scores)
            else:
                print("Analysis failed.")
    else:
        print("Model could not be loaded. Analysis skipped.")
    
    print("\nProgram finished.")
    
    
    