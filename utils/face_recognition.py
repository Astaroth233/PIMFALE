import cv2
import base64
import numpy as np
import dlib

# Initialize dlib's face detector, landmark predictor, and face recognition model
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognizer = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

def capture_face_data():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    face_img = None

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Face Capture', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if len(faces) > 0:
                x, y, w, h = faces[0]  # Capture the first detected face
                face_img = frame[y:y + h, x:x + w]
                break
        elif key == ord('c'):
            face_img = None
            break

    cap.release()
    cv2.destroyAllWindows()

    if face_img is not None:
        _, buffer = cv2.imencode('.jpg', face_img)
        face_data = base64.b64encode(buffer).decode('utf-8')
        return face_data
    return None

def calculate_similarity(stored_face_path, new_face_data):
    new_face_image = cv2.imdecode(np.frombuffer(base64.b64decode(new_face_data), np.uint8), cv2.IMREAD_COLOR)
    stored_face_image = cv2.imread(stored_face_path)
    
    stored_face_encodings = encode_faces(stored_face_image)
    new_face_encodings = encode_faces(new_face_image)

    print("Stored Face Encodings:", stored_face_encodings)
    print("New Face Encodings:", new_face_encodings)

    if stored_face_encodings is not None and new_face_encodings is not None:
        similarity = np.linalg.norm(stored_face_encodings - new_face_encodings)
        similarity_percentage = 1 - similarity  # Convert distance to similarity percentage
        print(f"Similarity Percentage: {similarity_percentage * 100:.2f}%")
        return similarity_percentage
    return 0

def encode_faces(image):
    detected_faces = face_detector(image, 1)
    if detected_faces:
        shape = shape_predictor(image, detected_faces[0])
        face_encoding = np.array(face_recognizer.compute_face_descriptor(image, shape))
        return face_encoding
    return None
