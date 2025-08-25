👤 AI Face Shape Predictor

An AI-powered real-time application that detects a human face from a live webcam feed, captures the cropped face, and classifies it into one of five shapes:
Heart | Oblong | Oval | Round | Square.

The app provides a smooth and user-friendly experience with a countdown timer, on-screen instructions, and a beautiful animated chart that visualizes prediction confidence.

✨ Key Features

🎥 Real-time Face Detection – Powered by MediaPipe, ensuring fast and accurate detection.

📸 Automated Face Capture – A high-quality cropped face is captured after a countdown.

🧠 Deep Learning Classification – Pre-trained TensorFlow/Keras model predicts the face shape.

📊 Interactive Visualization – Animated Matplotlib bar chart shows confidence scores dynamically.

🛠️ Tech Stack

Python

TensorFlow / Keras

OpenCV

MediaPipe

Matplotlib

🚀 How It Works

Start the application – The webcam feed is launched.

Face Detection – MediaPipe locates your face in real time.

Countdown & Capture – After a short timer, your face is automatically cropped.

Prediction – The AI model classifies your face into one of 5 categories.

Visualization – Animated bar chart displays confidence scores for each shape.

📂 Project Structure

AI-Face-Shape-Predictor/
│── real-time_analyzer.py                # Main application
│── train_face_classifier.ipynb          # Python notebook for training the model
|── preprocess_dataset.py                # Python script to preprocess the raw dataset.
│
│── models/
│   ├──  face_shape_classifier_5_classes.h5     (This is the best model, saved during training.)             
│   |──  blaze_face_short_range.tflite          (This is the mediapipe's face detection model.)
|
│── requirements.txt          # Dependencies
│── README.md                 # Project documentation

⚡ Installation

Clone the repository:

git clone https://github.com/Neeraj-0217/real-time-faceshape-predictor.git
cd real-time-faceshape-predictor


Create a virtual environment (optional but recommended):

python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate      # (Windows)


Install dependencies:

pip install -r requirements.txt

▶️ Usage

Run the main application:

python src/app.py


Ensure your webcam is connected.

Follow the on-screen instructions.

View your predicted face shape + confidence scores in a neat animated chart.

📊 Example Output

(Insert a demo GIF or screenshot here)

🔮 Future Improvements

Add more face shape categories (e.g., Diamond).

Improve model accuracy with a larger dataset.

Deploy as a web app or mobile app for broader accessibility.

🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to fork this repo and submit a pull request.

📜 License

This project is licensed under the MIT License.