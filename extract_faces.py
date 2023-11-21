import os
import face_recognition
import cv2


def extract_and_save_faces(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all image files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is an image (you may want to add more specific checks)
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            # Construct the full path of the input image
            input_image_path = os.path.join(input_folder, filename)

            # Load the input image
            input_image = face_recognition.load_image_file(input_image_path)

            # Find face locations in the image
            face_locations = face_recognition.face_locations(input_image)

            # Load the input image using OpenCV
            input_image_cv2 = cv2.imread(input_image_path)

            for i, face_location in enumerate(face_locations):
                # Extract the coordinates of the face location
                top, right, bottom, left = face_location

                # Crop the face from the input image
                face_image = input_image_cv2[top:bottom, left:right]

                # Create an output filename (e.g., face_1.jpg, face_2.jpg, etc.)
                output_filename = f"{filename.split('.')[0]}_face_{i + 1}.jpg"

                # Save the extracted face to the output folder
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, face_image)


if __name__ == "__main__":
    # Specify the input folder containing images
    input_folder = "C:\\Users\\Jarvis\\Desktop\\BTP_face\\#face_recognizer\\validation"

    # Specify the output folder where extracted faces will be saved
    output_folder = "faces_output"

    # Call the function to extract and save faces
    extract_and_save_faces(input_folder, output_folder)
