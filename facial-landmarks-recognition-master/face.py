# Import the necessary packages
from imutils import face_utils
import dlib
import cv2
import face_recognition
import pickle
import os
import csv
from datetime import datetime

# Initialize dlib's face detector (HOG-based) and the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# Initialize video capture from the default camera
cap = cv2.VideoCapture(0)

# Load known faces from file if it exists
if os.path.exists("known_faces.pkl"):
    with open("known_faces.pkl", "rb") as file:
        known_face_encodings, known_face_names = pickle.load(file)
else:
    known_face_encodings = []
    known_face_names = []

# Dictionary to keep track of recognized faces in the current session
recognized_faces = {}

def save_known_faces():
    """Save known faces and names to a file."""
    with open("known_faces.pkl", "wb") as file:
        pickle.dump((known_face_encodings, known_face_names), file)

def add_new_face(name):
    """Capture a new face and add it to known faces."""
    print(f"Scan your face for {name}...")
    while True:
        # Capture frame from the camera
        ret, image = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            (x, y, w, h) = face_utils.rect_to_bb(rect)
            face_img = image[y:y+h, x:x+w]
            rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_face)

            if face_encodings:
                face_encoding = face_encodings[0]
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)
                save_known_faces()
                print(f"Face of {name} has been saved.")
                return

        cv2.imshow("Scan Face", image)
        if cv2.waitKey(1) & 0xFF == 27:
            print("Face capture aborted.")
            break

def delete_face(name):
    """Delete a known face by name."""
    if name in known_face_names:
        index = known_face_names.index(name)
        del known_face_names[index]
        del known_face_encodings[index]
        save_known_faces()
        print(f"Face of {name} has been deleted.")
    else:
        print(f"No face found with the name {name}.")

def log_attendance(name, time):
    """Log attendance data to a CSV file."""
    with open("attendance_log.csv", "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, time])

def recognize_faces():
    """Recognize faces from the camera feed."""
    while True:
        # Capture frame from the camera
        ret, image = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            (x, y, w, h) = face_utils.rect_to_bb(rect)
            face_img = image[y:y+h, x:x+w]
            rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_face)

            # Check if any face encodings were found
            if face_encodings:
                # Take the first encoding found (assuming one face per frame)
                face_encoding = face_encodings[0]

                # Compare the found face encoding with known face encodings
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                
                # Initialize the display name as "Unknown"
                name_to_display = "Unknown"

                # Check if there is at least one match
                if True in matches:
                    # Find the index of the first matched face
                    first_match_index = matches.index(True)
                    
                    # Retrieve the corresponding name for the matched face
                    name_to_display = known_face_names[first_match_index]

                    # If the face hasn't been recognized in this session, log the time
                    if name_to_display not in recognized_faces:
                        login_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        recognized_faces[name_to_display] = login_time
                        print(f"{name_to_display} logged in at {login_time}")
                        log_attendance(name_to_display, login_time)

                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.rectangle(image, (x, y+h + 25), (x+w, y+h), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(image, name_to_display, (x + 6, y + h + 20), font, 0.5, (255, 255, 255), 1)

            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        cv2.imshow("Output", image)

        # Add new face on pressing 'n'
        k = cv2.waitKey(1) & 0xFF
        if k == ord('n'):
            name = input("Enter the name for the new face: ")
            add_new_face(name)
        elif k == ord('d'):
            name = input("Enter the name of the face to delete: ")
            delete_face(name)
        elif k == 27:  # ESC key
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    # Create CSV file with headers if it doesn't exist
    if not os.path.exists("attendance_log.csv"):
        with open("attendance_log.csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Login Time"])

    recognize_faces()
