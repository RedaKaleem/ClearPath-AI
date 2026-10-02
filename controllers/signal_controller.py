# controllers/signal_controller.py

def control_signal(emergency_detected, traffic_level):
    if emergency_detected:
        print("🚨 Emergency detected — GREEN light for emergency lane")
    elif traffic_level > 40:
        print("🔴 High traffic detected — extending GREEN duration")
    else:
        print("🟢 Normal signal operation")
