from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO(r"C:\runs\classify\train4\weights\best.pt")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to access camera")
        break

    # Predict emotion
    results = model(frame)

    probs = results[0].probs
    class_id = probs.top1
    confidence = probs.top1conf

    label = f"{model.names[class_id]} {confidence:.2f}"

    # Display prediction
    cv2.putText(frame, label, (20,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.imshow("Emotion Detection", frame)

    # press q to exit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()