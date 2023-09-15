import face_recognition
import cv2
import pickle
from datetime import datetime

encodings_location = "output/encodings_pavan.pkl"
with open(encodings_location, "rb") as f:
    loaded_encodings = pickle.load(f)

# Load known face encodings and names
known_encodings = loaded_encodings["encodings"]  # List of known face encodings
known_names = loaded_encodings["names"]  # List of corresponding names


# simple one...just storing start and end time
# def mark_db(name, action):
#     with open("output/tracking.csv", "a") as f:
#         now = datetime.now()
#         time = now.strftime("%H:%M:%S")
#         date = now.strftime("%Y-%m-%d")
#         f.writelines(f"\n{name},{action},{time},{date}")


def mark_db(name, action, tracking_start_time):
    with open("output/tracking.csv", "a") as f:
        now = datetime.now()
        time = now.strftime("%H:%M:%S")
        date = now.strftime("%Y-%m-%d")

        if action == "start":
            tracking_start_time = now
        elif action == "end":
            tracking_end_time = now
            tracking_duration = tracking_end_time - tracking_start_time
            hours, remainder = divmod(tracking_duration.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            formatted_duration = f"{hours:02}:{minutes:02}:{seconds:02}"
            f.writelines(
                f"\n{name}, {tracking_start_time.strftime('%H:%M:%S')}, {tracking_end_time.strftime('%H:%M:%S')}, {formatted_duration}, {date}"
            )

    return tracking_start_time


tracking_start_time = None

# Initialize video capture
cap = cv2.VideoCapture(0)

# Define how many frames to skip before processing
frames_to_skip = 0  # You can adjust this value to balance speed and accuracy
size_ratio = 1
tolerance = 0.45

tracking = False

cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)  # Create a named window
cv2.resizeWindow("Tracking", 800, 600)
is_fullscreen = False

# Initialize object tracker
# tracker = cv2.legacy.TrackerMIL_create()
tracker = cv2.legacy.TrackerKCF_create()
# tracker = cv2.legacy.TrackerTLD_create()
# tracker = cv2.legacy.TrackerMedianFlow_create()
# tracker = cv2.legacy.TrackerCSRT_create()
# tracker = cv2.legacy.TrackerMOSSE_create()

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

    if not tracking:
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
                face_distances = face_recognition.face_distance(
                    known_encodings, face_encoding
                )
                distance = face_distances[matched_index]
                print(distance)
                tracking_start_time = mark_db(name, "start", tracking_start_time)
                top, right, bottom, left = face_location
                tracking_bbox = (left, top, right - left, bottom - top)
                tracking = True
                tracker.init(frame, tracking_bbox)
                break  # Break out of the loop once a face is recognized

    else:
        success, tracking_bbox = tracker.update(frame)
        if success:
            # Tracking successful, draw the tracking box
            x, y, w, h = map(int, tracking_bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Draw the name at the bottom of the tracking box
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (x + 6, y + h + 20), font, 1, (0, 255, 0), 2)
        else:
            # Tracking lost, switch back to face recognition
            tracking = False
            mark_db(name, "end", tracking_start_time)
            tracking_start_time = None  # Reset tracking start time
            tracker = cv2.legacy.TrackerKCF_create()

    # print(tracking)
    # Display the frame
    cv2.imshow("Tracking", frame)

    # Toggle full-screen mode on 'F' key press
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("f"):
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty(
                "Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        else:
            cv2.setWindowProperty(
                "Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
            )

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
