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

window = tk.Tk()
window.geometry("480x900")  # Aumentamos la altura
window.title("Swoleboi - Detector de Sentadilla Búlgara")
ck.set_appearance_mode("dark")

classLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
classLabel.place(x=10, y=1)
classLabel.configure(text='ETAPA')
counterLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
counterLabel.place(x=160, y=1)
counterLabel.configure(text='REPS')
probLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
probLabel.place(x=300, y=1)
probLabel.configure(text='PROB')

classBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
classBox.place(x=10, y=41)
classBox.configure(text='0')
counterBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
counterBox.place(x=160, y=41)
counterBox.configure(text='0')
probBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
probBox.place(x=300, y=41)
probBox.configure(text='0')

legLabel = ck.CTkLabel(window, height=30, width=200, font=("Arial", 16), text_color="white", fg_color="green")
legLabel.place(x=10, y=85)
legLabel.configure(text='Pierna activa: --')

# Frame para la cámara - mantener posición original
frame = tk.Frame(height=480, width=480)
frame.place(x=10, y=120)
lmain = tk.Label(frame)
lmain.place(x=0, y=0)

# Mover todos los elementos debajo de la cámara (y=120+480=600)
angleLabel = ck.CTkLabel(window, height=30, width=400, font=("Arial", 12), text_color="white", fg_color="gray")
angleLabel.place(x=10, y=610)  # Movido hacia abajo
angleLabel.configure(text='Ángulos: --')

hipLabel = ck.CTkLabel(window, height=30, width=400, font=("Arial", 12), text_color="white", fg_color="gray")
hipLabel.place(x=10, y=645)  # Movido hacia abajo
hipLabel.configure(text='Cadera: --')

def reset_counter():
    global counter
    counter = 0

def reset_errors():
    """Resetea la visualización de errores"""
    global form_errors
    form_errors = []

button = ck.CTkButton(window, text='RESET REPS', command=reset_counter, height=40, width=120, font=("Arial", 16), text_color="white", fg_color="blue")
button.place(x=10, y=680)  # Movido hacia abajo

error_button = ck.CTkButton(window, text='RESET ERRORES', command=reset_errors, height=40, width=140, font=("Arial", 16), text_color="white", fg_color="red")
error_button.place(x=140, y=680)  # Movido hacia abajo

# Sección de análisis de forma
formLabel = ck.CTkLabel(window, height=30, width=200, font=("Arial", 16), text_color="black", padx=10)
formLabel.place(x=10, y=725)  # Movido hacia abajo
formLabel.configure(text='ANÁLISIS DE FORMA')

# Status box para forma
formStatusBox = ck.CTkLabel(window, height=30, width=200, font=("Arial", 14), text_color="white", fg_color="green")
formStatusBox.place(x=10, y=755)  # Movido hacia abajo
formStatusBox.configure(text='✓ FORMA CORRECTA')

# Error display label
errorLabel = ck.CTkLabel(window, height=30, width=250, font=("Arial", 16), text_color="black", padx=10)
errorLabel.place(x=220, y=725)  # Movido hacia abajo
errorLabel.configure(text='ERRORES DETECTADOS')

# Error box para mostrar errores
errorBox = ck.CTkTextbox(window, height=60, width=250, font=("Arial", 10))
errorBox.place(x=220, y=755)  # Movido hacia abajo

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

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

def analyze_bulgarian_squat_form(landmarks_data, active_leg, work_leg_angle, support_leg_angle, hip_movement):
    """
    Analiza la forma de la sentadilla búlgara y detecta errores comunes
    """
    errors = []
    
    if active_leg == 'none':
        return errors
    
    try:
        # Obtener puntos clave
        if active_leg == 'left':
            work_hip = [landmarks_data[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks_data[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            work_knee = [landmarks_data[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks_data[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            work_ankle = [landmarks_data[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks_data[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            support_hip = [landmarks_data[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                          landmarks_data[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            support_knee = [landmarks_data[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                           landmarks_data[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        else:
            work_hip = [landmarks_data[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks_data[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            work_knee = [landmarks_data[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks_data[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            work_ankle = [landmarks_data[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                         landmarks_data[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            support_hip = [landmarks_data[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                          landmarks_data[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            support_knee = [landmarks_data[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                           landmarks_data[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        
        # Obtener hombros para análisis de postura
        left_shoulder = [landmarks_data[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks_data[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks_data[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         landmarks_data[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        
        # Error 1: Rodilla muy hacia adelante (knee cave)
        if work_knee[0] > work_ankle[0] + 0.05:  # Rodilla muy adelantada
            errors.append({
                'severity': 'critical',
                'message': 'Rodilla muy adelante del pie'
            })
        
        # Error 2: Descenso insuficiente
        if work_leg_angle > 120 and hip_movement > 0.03:  # En posición baja pero ángulo muy abierto
            errors.append({
                'severity': 'moderate',
                'message': 'Descenso insuficiente'
            })
        
        # Error 3: Torso muy inclinado hacia adelante
        torso_lean = abs(left_shoulder[0] - right_shoulder[0])
        if torso_lean > 0.1:  # Hombros muy desalineados
            errors.append({
                'severity': 'moderate',
                'message': 'Torso muy inclinado'
            })
        
        # Error 4: Pie de apoyo sobrecargado
        if support_leg_angle < 160:  # Pierna de apoyo muy flexionada
            errors.append({
                'severity': 'critical',
                'message': 'Sobrecarga en pie de apoyo'
            })
        
        # Error 5: Cadera muy alta (no baja lo suficiente)
        if hip_movement < 0.02 and work_leg_angle < 130:  # Flexiona pero cadera no baja
            errors.append({
                'severity': 'moderate',
                'message': 'Cadera muy alta'
            })
        
        # Error 6: Rodilla colapsada hacia adentro
        hip_knee_distance = abs(work_hip[0] - work_knee[0])
        if hip_knee_distance < 0.02:  # Rodilla muy cerca de la línea de cadera
            errors.append({
                'severity': 'critical',
                'message': 'Rodilla colapsada hacia adentro'
            })
        
    except Exception as e:
        print(f"Error en análisis de forma: {e}")
    
    return errors

try:
    with open('bulgarian_squat.pkl', 'rb') as f:
        model = pickle.load(f)
    use_model = True
    print("Modelo de sentadilla búlgara cargado exitosamente")
except FileNotFoundError:
    print("Modelo 'bulgarian_squat.pkl' no encontrado. Usando detección por ángulos.")
    use_model = False

cap = cv2.VideoCapture(0)
current_stage = ''
counter = 0
bodylang_prob = np.array([0,0])
bodylang_class = ''
active_leg = 'none'  # 'left' o 'right'
hip_baseline = 0  # Posición inicial de cadera para comparar
form_errors = []  # Lista para almacenar errores de forma

def detect_active_leg(hip_left, hip_right, knee_left, knee_right, ankle_left, ankle_right):
    """
    Detecta cuál pierna está siendo usada como pierna de trabajo
    basándose en la posición y altura de las piernas
    """
    
    ankle_height_diff = ankle_left[1] - ankle_right[1]  # Diferencia de altura
    knee_position_diff = knee_left[0] - knee_right[0]   # Diferencia horizontal
        
    if abs(ankle_height_diff) > 0.15:  # Diferencia significativa de altura
        if ankle_height_diff < -0.15:  # Izquierdo más alto
            return 'right'  # Pierna derecha es la de trabajo
        elif ankle_height_diff > 0.15:  # Derecho más alto
            return 'left'   # Pierna izquierda es la de trabajo
    
    return 'none'

def detect():
    global current_stage
    global counter
    global bodylang_class
    global bodylang_prob
    global active_leg
    global hip_baseline
    global form_errors

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
            # Detección basada en ángulos y posición para sentadilla búlgara
            # Puntos clave
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
            
            # Calcular ángulos de las rodillas
            angle_left = calculate_angle(hip_left, knee_left, ankle_left)
            angle_right = calculate_angle(hip_right, knee_right, ankle_right)
            
            # Detectar pierna activa
            detected_leg = detect_active_leg(hip_left, hip_right, knee_left, knee_right, ankle_left, ankle_right)
            if detected_leg != 'none':
                active_leg = detected_leg
            
            # Calcular posición promedio de cadera para detectar movimiento vertical
            hip_center_y = (hip_left[1] + hip_right[1]) / 2
            
            # Establecer baseline de cadera si no existe
            if hip_baseline == 0:
                hip_baseline = hip_center_y
            
            # Calcular movimiento de cadera
            hip_movement = hip_center_y - hip_baseline
            
            # Seleccionar ángulo de la pierna de trabajo
            if active_leg == 'left':
                work_leg_angle = angle_left
                support_leg_angle = angle_right
            elif active_leg == 'right':
                work_leg_angle = angle_right
                support_leg_angle = angle_left
            else:
                work_leg_angle = min(angle_left, angle_right)  # Usar la más flexionada
                support_leg_angle = max(angle_left, angle_right)
            
            # ANÁLISIS DE FORMA - Detectar errores
            form_errors = analyze_bulgarian_squat_form(landmarks_data, active_leg, work_leg_angle, support_leg_angle, hip_movement)
            
            # Actualizar displays de error
            update_error_display(form_errors)
            
            # Actualizar labels para debugging
            angleLabel.configure(text=f'Trabajo: {work_leg_angle:.1f}° Soporte: {support_leg_angle:.1f}° Activa: {active_leg}')
            hipLabel.configure(text=f'Cadera Y: {hip_center_y:.3f} Baseline: {hip_baseline:.3f} Mov: {hip_movement:.3f}')
            legLabel.configure(text=f'Pierna activa: {active_leg.upper() if active_leg != "none" else "DETECTANDO..."}')
                        
            # Posición inicial: cadera alta, pierna de trabajo extendida
            initial_position = (work_leg_angle > 140 and hip_movement < 0.05)
            
            # Posición de trabajo: pierna flexionada, cadera baja
            work_position = (work_leg_angle < 110 and hip_movement > 0.08)
            
            # Detectar posición baja de sentadilla búlgara
            if work_position and active_leg != 'none':
                if current_stage != "abajo":
                    current_stage = "abajo"
                    bodylang_prob = np.array([0.85, 0.15])
                    bodylang_class = "down"
            
            # Detectar vuelta a posición inicial
            elif initial_position and active_leg != 'none':
                if current_stage == "abajo":
                    current_stage = "arriba"
                    counter += 1
                    bodylang_prob = np.array([0.15, 0.85])
                    bodylang_class = "up"
                elif current_stage == "":
                    current_stage = "inicio"
                    bodylang_prob = np.array([0.1, 0.9])
                    bodylang_class = "up"
            
            # Reset baseline si la persona cambia de posición significativamente
            if abs(hip_movement) > 0.2:
                hip_baseline = hip_center_y * 0.3 + hip_baseline * 0.7  # Suavizar el cambio

    except Exception as e:
        print(f"Error en detección: {e}")

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