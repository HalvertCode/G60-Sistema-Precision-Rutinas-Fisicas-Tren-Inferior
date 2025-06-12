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

print("Presiona 'u' para marcar posición ARRIBA (talones elevados)")
print("Presiona 'd' para marcar posición ABAJO (talones en el suelo)")
print("Presiona 'q' para salir")

while True:
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    if results.pose_landmarks:
        # Extraer landmarks
        row = np.array([[res.x, res.y, res.z, res.visibility] 
                       for res in results.pose_landmarks.landmark]).flatten().tolist()
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('u'):  # Posición arriba (talones elevados)
            row.append('up')
            data.append(row)
            print(f"Muestra 'up' guardada. Total: {len(data)}")
        elif key == ord('d'):  # Posición abajo (talones en el suelo)
            row.append('down')
            data.append(row)
            print(f"Muestra 'down' guardada. Total: {len(data)}")
        elif key == ord('q'):
            break
    
    cv2.imshow('Recolección de datos - Elevación de pantorrillas', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

cap.release()
cv2.destroyAllWindows()

# Guardar datos
df = pd.DataFrame(data, columns=landmarks + ['class'])
df.to_csv('calf_raises_data.csv', index=False)
print(f"Datos guardados: {len(data)} muestras")