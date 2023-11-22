# Facial Recognition Repository

This repository contains Python scripts for facial recognition tasks, offering a versatile set of functionalities for training, testing, validation,image/video processing and real-time recognition, with or without object tracking.

## Dataset format:

````face_recognizer/
│
├── output/
│
├── training/
│ └── class_1/
│ ├── img_1.jpg
│ └── img_2.png
│
├── validation/
│ ├── image1.jpg
│ └── image2.jpg
│
├── detector.py```

## Scripts Overview:

1. **Detector.py:**
   Detector.py is a versatile script that can perform various facial recognition tasks:

   - **Training and Encoding:** Train the model on input data and save encodings for future use.
   - **Testing:** Test on a single image to recognize faces.
   - **Validation:** Validate on multiple images and save processed images with recognized faces.
   - **Live Recognition:** Perform real-time face recognition using a webcam or video feed.
   - **Evaluate:** Evaluate the model on the test dataset.

2. **Lite.py:**
   Lite.py is a simplified version of Detector.py, focusing primarily on real-time recognition. It loads pre-trained encodings and attempts to match faces in a live video feed.

3. **Tracking.py:**
   Tracking.py is a script with two main phases:

   - **Face Recognition Phase:** Recognize faces and, when a match is found, initiate tracking.
   - **Object Tracking Phase:** Track recognized faces and store tracking data in a database file (CSV).

4. **Lite_tracking.py:**
   Lite_tracking.py is a more streamlined version of Tracking.py, emphasizing live recognition and tracking.

5. **Video_process.py:**
   Video_process.py is similar to Detector.py but specifically designed for processing videos. It performs recognition on video frames and saves the processed video output.

6. **Video_process_tracking.py:**
   Video_process_tracking.py is similar to Tracking.py but tailored for video processing. It combines recognition and tracking on video frames and saves the processed video output.

## Dependencies:

Ensure you have the following dependencies installed:

- Python (>=3.6)
- OpenCV (cv2)
- face_recognition
- Other relevant libraries (specified in individual script requirements)

Feel free to explore each script for detailed instructions and usage.
````
