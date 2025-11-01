import cv2
import serial
import time

# Initialize serial
try:
    ser = serial.Serial('COM3', 9600, timeout=1)  # Change COM port
    time.sleep(2)
except serial.SerialException:
    print("Error: Could not open serial port.")
    exit()

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Lower resolution for speed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Control parameters
KP = 0.05
DEADZONE = 20  # Smaller deadzone for snappier tracking
MIN_ANGLE, MAX_ANGLE = 30, 150
current_angle = 90  # Start centered

ret, frame = cap.read()
if not ret:
    print("Error: Could not open webcam.")
    ser.close()
    exit()

frame_center_x = frame.shape[1] // 2
prev_cx, prev_cy = None, None
prev_time = time.time()
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    # Process every frame (or skip for performance)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    current_time = time.time()
    time_diff = current_time - prev_time if prev_time else 1.0

    direction = ""
    speed = 0.0
    motor_adjusted = False

    for (x, y, w, h) in faces:
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Movement tracking
        if prev_cx is not None and prev_cy is not None:
            dx = cx - prev_cx
            dy = cy - prev_cy
            if abs(dx) > 10 or abs(dy) > 10:
                if abs(dx) > abs(dy):
                    direction = "Right" if dx > 0 else "Left"
                else:
                    direction = "Down" if dy > 0 else "Up"
            distance = ((dx ** 2) + (dy ** 2)) ** 0.5
            speed = distance / time_diff if time_diff > 0 else 0

        # Stepper control (only send if significant change)
        offset_x = cx - frame_center_x
        if abs(offset_x) > DEADZONE:
            angle_adjust = round(KP * offset_x)
            new_angle = current_angle - angle_adjust
            new_angle = max(MIN_ANGLE, min(MAX_ANGLE, new_angle))
            if abs(new_angle - current_angle) >= 1:
                ser.write(f"{new_angle}\n".encode())
                current_angle = new_angle
                motor_adjusted = True

        prev_cx, prev_cy = cx, cy
        prev_time = current_time
        break  # Track first face only

    # Display info
    text = f"Direction: {direction}, Speed: {speed:.2f}px/s"
    if motor_adjusted:
        text += f" | Stepper: {current_angle}°"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Face Stepper Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
ser.close()
cap.release()
cv2.destroyAllWindows()
