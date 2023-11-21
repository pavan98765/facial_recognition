import face_recognition
import cv2
import pickle

encodings_location = "output/encodings.pkl"
with open(encodings_location, "rb") as f:
    loaded_encodings = pickle.load(f)

# Load known face encodings and names
known_encodings = loaded_encodings["encodings"]  # List of known face encodings
known_names = loaded_encodings["names"]  # List of corresponding names

# Initialize video capture
cap = cv2.VideoCapture(0)

# Define how many frames to skip before processing
frames_to_skip = 0  # You can adjust this value to balance speed and accuracy
size_ratio = 1
tolerance = 0.6

frame_count = 0
while True:
    # Skip frames in the video
    for _ in range(frames_to_skip):
        ret, _ = cap.read()
        if not ret:
            break

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=size_ratio, fy=size_ratio)

    # Find face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare face encoding with known encodings
        matches = face_recognition.compare_faces(
            known_encodings, face_encoding, tolerance=tolerance
        )
        # Check if any match is found
        if any(matches):
            matched_index = matches.index(True)
            name = known_names[matched_index]
            # Draw bounding box and name on the frame for known faces only
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1
            )

    # Display the frame
    cv2.imshow("Real-time Face Recognition", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
