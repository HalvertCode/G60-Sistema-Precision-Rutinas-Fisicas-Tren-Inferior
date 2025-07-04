import tkinter as tk
import customtkinter as ck
import pandas as pd
import numpy as np
import pickle
import mediapipe as mp
import cv2
from PIL import Image, ImageTk
from landmarks import landmarks
import math
from lunges_form_checker import LungesFormChecker

window = tk.Tk()
window.geometry("480x850")  # Aumentar altura de ventana
window.title("Swoleboi - Detector de Zancadas")
ck.set_appearance_mode("dark")

# Labels de información
classLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
classLabel.place(x=10, y=1)
classLabel.configure(text='ETAPA')
counterLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
counterLabel.place(x=160, y=1)
counterLabel.configure(text='REPS')
probLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
probLabel.place(x=300, y=1)
probLabel.configure(text='PROB')

# Cajas de información
classBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
classBox.place(x=10, y=41)
classBox.configure(text='0')
counterBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
counterBox.place(x=160, y=41)
counterBox.configure(text='0')
probBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
probBox.place(x=300, y=41)
probBox.configure(text='0')

# Frame para video
frame = tk.Frame(height=480, width=480)
frame.place(x=10, y=90)
lmain = tk.Label(frame)
lmain.place(x=0, y=0)

# Label para mostrar ángulos (debugging) - movido debajo del video
angleLabel = ck.CTkLabel(window, height=50, width=460, font=("Arial", 12), text_color="white", fg_color="gray")
angleLabel.place(x=10, y=580)  # Justo debajo del video
angleLabel.configure(text='Estado: Esperando posición...')

# Botones de reset - movidos más abajo
def reset_counter():
    global counter
    counter = 0

def reset_errors():
    global form_checker
    form_checker.clear_errors()
    update_error_display([])

button = ck.CTkButton(window, text='RESET REPS', command=reset_counter, height=40, width=120, font=("Arial", 16), text_color="white", fg_color="blue")
button.place(x=10, y=640)  # Movido de 600 a 640

error_reset_button = ck.CTkButton(window, text='RESET ERRORES', command=reset_errors, height=40, width=120, font=("Arial", 16), text_color="white", fg_color="red")
error_reset_button.place(x=150, y=640)  # Movido de 600 a 640

# Labels para análisis de forma - movidos más abajo
formLabel = ck.CTkLabel(window, height=30, width=200, font=("Arial", 16), text_color="black", padx=10)
formLabel.place(x=10, y=690)  # Movido de 650 a 690
formLabel.configure(text='ANÁLISIS DE FORMA')

errorsLabel = ck.CTkLabel(window, height=30, width=200, font=("Arial", 16), text_color="black", padx=10)
errorsLabel.place(x=250, y=690)  # Movido de 650 a 690
errorsLabel.configure(text='ERRORES DETECTADOS')

# Cajas para mostrar estado de forma y errores - movidas más abajo
formStatusBox = ck.CTkLabel(window, height=40, width=200, font=("Arial", 16), text_color="white", fg_color="green")
formStatusBox.place(x=10, y=720)  # Movido de 680 a 720
formStatusBox.configure(text="✓ FORMA CORRECTA")

errorBox = ck.CTkTextbox(window, height=80, width=200, font=("Arial", 12))
errorBox.place(x=250, y=720)  # Movido de 680 a 720

# Configuración de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

# Inicializar el checker de forma
form_checker = LungesFormChecker()

# Función para calcular ángulos
def calculate_angle(a, b, c):
    """
    Calcula el ángulo entre tres puntos
    a, b, c son arrays numpy con coordenadas [x, y]
    """
    a = np.array(a)
    b = np.array(b)  # vértice
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def is_proper_lunge(hip_left, knee_left, ankle_left, hip_right, knee_right, ankle_right):
    """
    Verifica si la posición actual es una zancada correcta:
    - Ambas rodillas entre 80-90°
    - Una rodilla apuntando hacia adelante (pierna delantera)
    - Una rodilla apuntando hacia abajo (pierna trasera)
    """
    # Calcular ángulos de las rodillas
    angle_left = calculate_angle(hip_left, knee_left, ankle_left)
    angle_right = calculate_angle(hip_right, knee_right, ankle_right)
    
    # Verificar que ambos ángulos estén en el rango correcto (80-90°)
    angles_in_range = (80 <= angle_left <= 90) and (80 <= angle_right <= 90)
    
    if not angles_in_range:
        return False, angle_left, angle_right, "Ángulos fuera de rango"
    
    # Determinar orientación de las rodillas
    left_knee_y = knee_left[1]
    right_knee_y = knee_right[1]
    left_knee_x = knee_left[0]
    right_knee_x = knee_right[0]
    
    # Calcular vectores para determinar orientación de las rodillas
    left_thigh_vector = np.array([knee_left[0] - hip_left[0], knee_left[1] - hip_left[1]])
    left_shin_vector = np.array([ankle_left[0] - knee_left[0], ankle_left[1] - knee_left[1]])
    
    right_thigh_vector = np.array([knee_right[0] - hip_right[0], knee_right[1] - hip_right[1]])
    right_shin_vector = np.array([ankle_right[0] - knee_right[0], ankle_right[1] - knee_right[1]])
    
    # Verificar orientación
    left_pointing_down = left_shin_vector[1] > 0.05
    right_pointing_down = right_shin_vector[1] > 0.05
    
    left_pointing_forward = abs(left_shin_vector[1]) < 0.05
    right_pointing_forward = abs(right_shin_vector[1]) < 0.05
    
    # Para una zancada correcta: una pierna apunta hacia adelante, otra hacia abajo
    proper_orientation = (left_pointing_forward and right_pointing_down) or (right_pointing_forward and left_pointing_down)
    
    # Información adicional para debugging
    status_info = f"Izq: {angle_left:.1f}° ({'↓' if left_pointing_down else '→' if left_pointing_forward else '?'}) "
    status_info += f"Der: {angle_right:.1f}° ({'↓' if right_pointing_down else '→' if right_pointing_forward else '?'})"
    
    if not proper_orientation:
        status_info += " - Orientación incorrecta"
    else:
        status_info += " - ¡Zancada correcta!"
    
    return proper_orientation, angle_left, angle_right, status_info

def update_error_display(form_errors):
    """Actualiza la visualización de errores en la interfaz"""
    if not form_errors:
        formStatusBox.configure(text="✓ FORMA CORRECTA", fg_color="green")
        errorBox.delete("0.0", "end")
        errorBox.insert("0.0", "Sin errores detectados")
    else:
        # Determinar color según severidad
        critical_errors = [e for e in form_errors if e['severity'] == 'critical']
        
        if critical_errors:
            formStatusBox.configure(text="⚠ FORMA CRÍTICA", fg_color="red")
        else:
            formStatusBox.configure(text="⚠ FORMA MEJORABLE", fg_color="orange")
        
        # Mostrar errores
        errorBox.delete("0.0", "end")
        for i, error in enumerate(form_errors[:3]):  # Mostrar máximo 3 errores
            severity_symbol = "🔴" if error['severity'] == 'critical' else "🟡"
            errorBox.insert("end", f"{severity_symbol} {error['message']}\n")

# Intentar cargar el modelo de zancadas (si existe)
try:
    with open('lunges.pkl', 'rb') as f:
        model = pickle.load(f)
    use_model = True
    print("Modelo de zancadas cargado exitosamente")
except FileNotFoundError:
    print("Modelo 'lunges.pkl' no encontrado. Usando detección mejorada por ángulos.")
    use_model = False

cap = cv2.VideoCapture(0)
current_stage = ''
counter = 0
bodylang_prob = np.array([0,0])
bodylang_class = ''

def detect():
    global current_stage
    global counter
    global bodylang_class
    global bodylang_prob

    ret, frame = cap.read()
    
    # Convertir BGR a RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    # Dibujar landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(106,13,173), thickness=4, circle_radius=5),
        mp_drawing.DrawingSpec(color=(255,102,0), thickness=5, circle_radius=10))

    try:
        landmarks_data = results.pose_landmarks.landmark
        
        # Extraer puntos clave
        hip_left = [landmarks_data[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks_data[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee_left = [landmarks_data[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks_data[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle_left = [landmarks_data[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks_data[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        hip_right = [landmarks_data[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    landmarks_data[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee_right = [landmarks_data[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                     landmarks_data[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle_right = [landmarks_data[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                      landmarks_data[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        
        # Obtener puntos adicionales para análisis de forma
        shoulder_left = [landmarks_data[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks_data[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        shoulder_right = [landmarks_data[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         landmarks_data[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        
        # Analizar forma con el form checker
        landmarks_dict = {
            'hip_left': hip_left,
            'knee_left': knee_left,
            'ankle_left': ankle_left,
            'hip_right': hip_right,
            'knee_right': knee_right,
            'ankle_right': ankle_right,
            'shoulder_left': shoulder_left,
            'shoulder_right': shoulder_right
        }
        
        form_errors = form_checker.analyze_form(landmarks_dict)
        update_error_display(form_errors)
        
        if use_model:
            # Usar modelo entrenado si está disponible
            row = np.array([[res.x, res.y, res.z, res.visibility] for res in landmarks_data]).flatten().tolist()
            X = pd.DataFrame([row], columns=landmarks)
            bodylang_prob = model.predict_proba(X)[0]
            bodylang_class = model.predict(X)[0]

            if bodylang_class == "down" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
                current_stage = "abajo"
            elif current_stage == "abajo" and bodylang_class == "up" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
                current_stage = "arriba"
                counter += 1
        else:
            # Verificar si es una zancada correcta
            is_lunge, angle_left, angle_right, status_info = is_proper_lunge(
                hip_left, knee_left, ankle_left, 
                hip_right, knee_right, ankle_right
            )
            
            # Actualizar label de estado
            angleLabel.configure(text=status_info)
            
            # Lógica mejorada para detectar zancadas
            if is_lunge:
                if current_stage != "abajo":
                    current_stage = "abajo"
                    bodylang_prob = np.array([0.9, 0.1])
                    bodylang_class = "down"
            else:
                # Detectar vuelta a posición inicial
                if angle_left > 160 and angle_right > 160:
                    if current_stage == "abajo":
                        current_stage = "arriba"
                        counter += 1
                        bodylang_prob = np.array([0.1, 0.9])
                        bodylang_class = "up"

    except Exception as e:
        print(f"Error en detección: {e}")
        angleLabel.configure(text=f"Error: {str(e)}")

    # Mostrar imagen
    img = image[:, :460, :]
    imgarr = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(imgarr)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, detect)

    # Actualizar interfaz
    counterBox.configure(text=counter)
    if len(bodylang_prob) > 0:
        probBox.configure(text=f'{bodylang_prob[bodylang_prob.argmax()]:.2f}')
    classBox.configure(text=current_stage)

detect()
window.mainloop()