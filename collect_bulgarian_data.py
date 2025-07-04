import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from landmarks import landmarks

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
data = []

print("=== RECOLECCIÓN DE DATOS - SENTADILLA BÚLGARA ===")
print("Presiona 'u' para marcar posición ARRIBA (inicial)")
print("Presiona 'd' para marcar posición ABAJO (trabajo)")
print("Presiona 'l' para cambiar a pierna IZQUIERDA")
print("Presiona 'r' para cambiar a pierna DERECHA")
print("Presiona 'q' para salir")

current_leg = "left"

while True:
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    # Mostrar información en pantalla
    cv2.putText(frame, f"Pierna activa: {current_leg.upper()}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Muestras: {len(data)}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    if results.pose_landmarks:
        row = np.array([[res.x, res.y, res.z, res.visibility] 
                       for res in results.pose_landmarks.landmark]).flatten().tolist()
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('u'):  # Posición arriba
            row.extend(['up', current_leg])
            data.append(row)
            print(f"Muestra 'up' - {current_leg} guardada. Total: {len(data)}")
        elif key == ord('d'):  # Posición abajo
            row.extend(['down', current_leg])
            data.append(row)
            print(f"Muestra 'down' - {current_leg} guardada. Total: {len(data)}")
        elif key == ord('l'):  # Cambiar a pierna izquierda
            current_leg = "left"
            print("Cambiado a pierna IZQUIERDA")
        elif key == ord('r'):  # Cambiar a pierna derecha
            current_leg = "right"
            print("Cambiado a pierna DERECHA")
        elif key == ord('q'):
            break
    
    cv2.imshow('Recolección - Sentadilla Búlgara', frame)

cap.release()
cv2.destroyAllWindows()

# Guardar datos
columns = landmarks + ['class', 'active_leg']
df = pd.DataFrame(data, columns=columns)
df.to_csv('bulgarian_squat_data.csv', index=False)
print(f"Datos guardados: {len(data)} muestras")
print(f"Distribución por pierna:")
print(df['active_leg'].value_counts())