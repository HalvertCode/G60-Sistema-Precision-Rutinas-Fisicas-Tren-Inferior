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

print("Presiona 'u' para marcar POSICIÓN ARRIBA (de pie)")
print("Presiona 'd' para marcar POSICIÓN ABAJO (en sentadilla)")
print("Presiona 'q' para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    if results.pose_landmarks:
        # Extraer landmarks
        row = np.array([[res.x, res.y, res.z, res.visibility] 
                        for res in results.pose_landmarks.landmark]).flatten().tolist()
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('u'):  # Posición arriba (de pie)
            row.append('up')
            data.append(row)
            print(f"Muestra 'up' guardada. Total: {len(data)}")
        elif key == ord('d'):  # Posición abajo (en sentadilla)
            row.append('down')
            data.append(row)
            print(f"Muestra 'down' guardada. Total: {len(data)}")
        elif key == ord('q'):
            break
    
    cv2.imshow('Recolección de datos - Squats', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

cap.release()
cv2.destroyAllWindows()

# Guardar datos
df = pd.DataFrame(data, columns=landmarks + ['class'])
df.to_csv('squats_data.csv', index=False)
print(f"Datos guardados: {len(data)} muestras")