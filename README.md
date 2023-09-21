# facial_recognition
Detector.py: It can train and save encodings for future use, test on a single image, validate on multiple images and can run on live web cam or video.
Lite.py: This is simpler version of detector.py focoused mainly on realtime recognition, it just loads the encodings and tries to match faces.
Tracking.py: Here the code has Two phases- Face Recognition Phase , Object Tracking Phase. It can recognize and when a match is found, does tracking , and stores all tracking data in a database file(csv).
Lite_tracking.py: Again a simpler version of tracking.py , just focouses on live recognition and tracking.
Video_process.py: It is detector.py , which only does recognition on videos and saves the processed video output
Video_process_tracking.py: it is tracking.py, which does recognition and tracking on videos and saves the processed video ouput.
