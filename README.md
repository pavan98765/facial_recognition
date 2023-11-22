# Facial Recognition Repository

This repository contains Python scripts for facial recognition tasks, offering a versatile set of functionalities for training, testing, validation,image/video processing and real-time recognition, with or without object tracking.

## Dataset format:

```
face_recognizer/
│
├── output/
│
├── training/
│ └── class1/
│       ├── img_1.jpg
│       └── img_2.png
│ └── class2/
│       ├── img_1.jpg
│       └── img_2.png
|
├── test/
│ └── class_name/
│       ├── img_1.jpg
│       └── img_2.png
│
├── validation/
│       ├── image1.jpg
│       └── image2.jpg
│
├── detector.py
├── image_process.py
└── video_process.py
```

## Scripts Overview:

1. **Detector.py:**
   Detector.py is a versatile script that can perform various facial recognition tasks:

   - **Training and Encoding:** Train the model on input data and save encodings for future use.Put the training data in training folder with the above format then run the code below:

```
python Detector.py --train
```

- **Testing:** Test on a single image to recognize faces.
- **Validation:** Validate on multiple images and save processed images with recognized faces.
- **Live Recognition:** Perform real-time face recognition using a webcam or video feed.
- **Evaluate:** Evaluate the model on the test dataset.

2. **image_process.py:**
   image_process.py takes in the images from the validation folder, predicts on it and stores them in the output folder.

3. **video_process.py:**
   video_process.py take in a video input and then predicts and stores the output in the output folder. this only does facial recognition on the frames.
4. **video_process_tracking.py:**
   video_process_tracking.py is a faster and better version of video_process.py, where it waits for a match and switches to tracking,later when the tracking is lost back to recogntion. Basically two phases tracking and recognition. It is faster and more suited for tracking a single person.

5. **Lite.py:**
   Lite.py is a simplified version of Detector.py, focusing primarily on real-time recognition. It loads pre-trained encodings and attempts to match faces in a live video feed.

6. **Tracking.py:**
   Tracking.py is a script with two main phases:

   - **Face Recognition Phase:** Recognize faces and, when a match is found, initiate tracking.
   - **Object Tracking Phase:** Track recognized faces and store tracking data in a database file (CSV).

7. **Lite_tracking.py:**
   Lite_tracking.py is a more streamlined version of Tracking.py, emphasizing live recognition and tracking.

8. **Video_process.py:**
   Video_process.py is similar to Detector.py but specifically designed for processing videos. It performs recognition on video frames and saves the processed video output.

9. **Video_process_tracking.py:**
   Video_process_tracking.py is similar to Tracking.py but tailored for video processing. It combines recognition and tracking on video frames and saves the processed video output.

## Dependencies:

Ensure you have the following dependencies installed:

- Python (>=3.6)
- OpenCV (cv2) == 4.6.0.66
- face_recognition

Feel free to explore each script for detailed instructions and usage.

```

```
