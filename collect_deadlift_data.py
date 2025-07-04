import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from landmarks import landmarks

# Configurar MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
data = []

print("Presiona 'u' para marcar posición ARRIBA (deadlift)")
print("Presiona 'd' para marcar posición ABAJO (deadlift)")
print("Presiona 'q' para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    if results.pose_landmarks:
        # Extraer landmarks
        row = np.array([[lm.x, lm.y, lm.z, lm.visibility] 
                        for lm in results.pose_landmarks.landmark]).flatten().tolist()
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('u'):  # Posición arriba del deadlift
            row.append('up')
            data.append(row)
            print(f"Muestra 'up' guardada. Total: {len(data)}")
        elif key == ord('d'):  # Posición abajo del deadlift
            row.append('down')
            data.append(row)
            print(f"Muestra 'down' guardada. Total: {len(data)}")
        elif key == ord('q'):
            break
    
    cv2.imshow('Recolección de datos - Deadlift', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

cap.release()
cv2.destroyAllWindows()

# Guardar datos
df = pd.DataFrame(data, columns=landmarks + ['class'])
df.to_csv('deadlift_data.csv', index=False)
print(f"Datos guardados: {len(data)} muestras")