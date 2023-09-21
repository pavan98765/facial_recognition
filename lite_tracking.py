import face_recognition
import cv2
import pickle

encodings_location = "output/encodings_pavan.pkl"
with open(encodings_location, "rb") as f:
    loaded_encodings = pickle.load(f)

known_encodings = loaded_encodings["encodings"]
known_names = loaded_encodings["names"]

cap = cv2.VideoCapture(0)

frames_to_skip = 0
size_ratio = 1
tolerance = 0.5

tracking = False
tracker = cv2.legacy.TrackerKCF_create()
while True:
    for _ in range(frames_to_skip):
        ret, _ = cap.read()
        if not ret:
            break

    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, None, fx=size_ratio, fy=size_ratio)

    if not tracking:
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(
                known_encodings, face_encoding, tolerance=tolerance
            )

            if any(matches):
                matched_index = matches.index(True)
                name = known_names[matched_index]
                top, right, bottom, left = face_location
                tracking_bbox = (left, top, right - left, bottom - top)
                tracking = True
                tracker.init(frame, tracking_bbox)
                break
    else:
        success, tracking_bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int, tracking_bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (x + 6, y + h + 20), font, 1, (0, 255, 0), 2)
        else:
            tracking = False
            tracker = cv2.legacy.TrackerKCF_create()

    # print(tracking)

    cv2.imshow("Face Recognition and Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
