import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def analyze_form(landmarks, active_leg):
    errors = []
    
    try:
        if active_leg == 'left':
            work_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            work_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            work_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        else:
            work_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            work_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            work_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # Detectar errores comunes
        knee_angle = calculate_angle(work_hip, work_knee, work_ankle)
        
        # Error: Rodilla adelantada
        if work_knee[0] > work_ankle[0] + 0.05:
            errors.append("Rodilla adelantada")
            
        # Error: Cadera alta
        if knee_angle < 100 and work_hip[1] < 0.8:
            errors.append("Cadera alta")
            
    except:
        pass
    
    return errors

def get_form_status(errors):
    if not errors:
        return "CORRECTO", (0, 255, 0)  # Verde
    elif any("Cadera alta" in e for e in errors):
        return "MEJORABLE", (0, 165, 255)  # Naranja
    else:
        return "CRÍTICO", (0, 0, 255)  # Rojo

# Ejemplo de uso
cap = cv2.VideoCapture(0)
active_leg = 'left'  # Cambiar según detección

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Procesamiento de pose
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        errors = analyze_form(landmarks, active_leg)
        status, color = get_form_status(errors)
        
        # Mostrar estado en pantalla
        cv2.putText(frame, f"Estado: {status}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Mostrar errores
        for i, error in enumerate(errors[:3]):
            cv2.putText(frame, error, (10, 70 + i*40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    
    cv2.imshow('Bulgarian Squat Form Checker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()