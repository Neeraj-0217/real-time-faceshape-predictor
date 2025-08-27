# 👤 Real-time FaceShape Predictor

> An AI-powered real-time application that detects a human face from a live webcam feed, captures the cropped face, and classifies it into one of five shapes:
Heart | Oblong | Oval | Round | Square.

The app provides a smooth and user-friendly experience with a countdown timer, on-screen instructions, and a beautiful animated chart that visualizes prediction confidence.

---

## ✨ Key Features

- 🎥 Real-time Face Detection – Powered by MediaPipe, ensuring fast and accurate detection.

- 📸 Automated Face Capture – A high-quality cropped face is captured after a countdown.

- 🧠 Deep Learning Classification – Pre-trained TensorFlow/Keras model predicts the face shape.

- 📊 Interactive Visualization – Animated Matplotlib bar chart shows confidence scores dynamically.

---

## 🛠️ Tech Stack

- Python

- TensorFlow / Keras

- OpenCV

- MediaPipe

- Matplotlib

---

## 🚀 How It Works

```text
Start the application – The webcam feed is launched.

Face Detection – MediaPipe locates your face in real time.

Countdown & Capture – After a short timer, your face is automatically cropped.

Prediction – The AI model classifies your face into one of 5 categories.

Visualization – Animated bar chart displays confidence scores for each shape.
````

---

## External Files & Dataset
* Face Shape Dataset - The datset used to train the classifier.
  - Download Link: https://storage.googleapis.com/kaggle-data-sets/463280/879701/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250825%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250825T142736Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=5c33c29d5531ffebd1b2a49a2eddbc4215ed461fe2b024651fb6ee9d643ee06ae6b873ebb00331073a2edc7059f9ed480555b9049eaa6517718de16ff2e102cf35bcce125784a0f2795f2d72469aa1cb91947a0049c917caea29bffa6cc869a95b630a522ccf617694038f473dd6c7bac1545696f509ffe0c1cff94849aab5df1ab7bf9d8d4f9323cb40f60c69ca4908053054d6144102de9cbf2a6613092e4fab62b6a6ea0573b571a0dd63b90ca85daaa229d88bb81c24da6c1f21a2dbcb3f968c34e037c9ba5d119adb3b313ff99f42005768f907185c08ffdffc6745b56b253ad4d3e2d61ad2fd1a9b47a34bf0c30f79cfcde42bf079fa099e7bfdd98ce7
* BlazeFace Short-Range Model - A pre-trained TFLite model for face detection.
  - Download Link: https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite
---

## 📂 Project Structure

```
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
```

---

## ⚡ Installation

* Clone the repository:
```
git clone https://github.com/Neeraj-0217/real-time-faceshape-predictor.git
cd real-time-faceshape-predictor
```

* Create a virtual environment (optional but recommended):
```
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate      # (Windows)
```

* Install dependencies:
```
pip install -r requirements.txt
```
▶️ Usage

* Run the main application:
```
python real-time_analyzer.py
```
  - Ensure your webcam is connected.

  - Follow the on-screen instructions.

  - View your predicted face shape + confidence scores in a neat animated chart.


* 📊 Example Output

<img width="800" height="636" alt="Screenshot 2025-08-25 193903" src="https://github.com/user-attachments/assets/431bf746-0b01-4e0d-9c6d-68f2a02f2f4d" />


<img width="1375" height="833" alt="Screenshot 2025-08-25 193254" src="https://github.com/user-attachments/assets/26ec4047-14c2-4858-8817-b5cf30d87a0f" />

--- 

## 📊 Model Performance
The pre-trained model's performance metrics on the validation dataset are as follows:
 - Accuracy: **0.6150**
 - Loss: **1.3057**

## ⚠️ Known Limitations
  - The model was trained primarily on a dataset of female faces. The accuracy for male faces may be lower due to this gender-imbalanced dataset.
  - The model's performance is not fully optimized. During development and training, I was limited by my system's physical specifications. As a result, the model's architecture was kept lightweight to ensure it could be trained successfully, and it may not achieve the same state-of-the-art results as models trained on more powerful hardware.

---

## 🔮 Future Improvements

* [ ] Add more face shape categories (e.g., Diamond).

* [ ] Improve model accuracy with a larger dataset.

* [ ] Deploy as a web app or mobile app for broader accessibility.

---

## 🙋‍♂️ About Me

I'm **Neeraj**, a student from India passionate about machine learning and computer vision.

📧 Email: [www.asneeraj@gmail.com](mailto:asneeraj@gmail.com) 

📌 This project is a part of my AI/ML journey — feedback and forks are welcome!

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! 
Feel free to fork this repo and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---
