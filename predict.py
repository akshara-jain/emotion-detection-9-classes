import cv2
from ultralytics import YOLO

# Load trained emotion model
model = YOLO(r"C:\runs\classify\train4\weights\best.pt")
# Open webcam
cap = cv2.VideoCapture(0)

person_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Reset person counter
    person_id = 1

    # Detect faces using Haarcascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Loop through all detected faces
    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        # Predict emotion
        results = model(face)

        # Get predicted label
        label = results[0].names[results[0].probs.top1]

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # Show Person number + Emotion
        text = f"Person {person_id}: {label}"

        cv2.putText(
            frame,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        person_id += 1

    cv2.imshow("Emotion Detection", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()