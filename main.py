import cv2
from controllers.detection import detect_emergency_vehicles
from controllers.signal_controller import control_signal

# Load three video feeds
cap1 = cv2.VideoCapture("video_feed/13028093_3840_2160_30fps.mp4")
cap2 = cv2.VideoCapture("video_feed/13105476_3840_2160_30fps.mp4")
cap3 = cv2.VideoCapture("video_feed/13527819_1080_1920_30fps.mp4")
cap4 = cv2.VideoCapture("video_feed/gettyimages-1460800864-640_adpp.mp4")

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()

    if not (ret1 and ret2 and ret3 and ret4):
        break

    # Emergency detection for each junction
    e1 = detect_emergency_vehicles(frame1)
    e2 = detect_emergency_vehicles(frame2)
    e3 = detect_emergency_vehicles(frame3)
    e4 = detect_emergency_vehicles(frame4)

    # Simulated traffic levels
    traffic1, traffic2, traffic3, traffic4 = 45, 60, 25, 55

    # Control logic for each signal
    print("\n--- JUNCTION 1 ---")
    control_signal(e1, traffic1)

    print("--- JUNCTION 2 ---")
    control_signal(e2, traffic2)

    print("--- JUNCTION 3 ---")
    control_signal(e3, traffic3)

    print("--- JUNCTION 4 ---")
    control_signal(e4, traffic4)

    # Display windows side-by-side
    cv2.imshow("Junction 1", frame1)
    cv2.imshow("Junction 2", frame2)
    cv2.imshow("Junction 3", frame3)
    cv2.imshow("Junction 4", frame4)
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release all
cap1.release()
cap2.release()
cap3.release()
cap4.release()
cv2.destroyAllWindows()
