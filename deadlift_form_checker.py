import numpy as np
import math
from collections import deque

class DeadliftFormChecker:
    def __init__(self, history_size=10):
        """
        Inicializa el verificador de forma del deadlift
        
        Args:
            history_size: Número de frames a mantener en historial para análisis
        """
        self.history_size = history_size
        self.angle_history = deque(maxlen=history_size)
        self.error_history = deque(maxlen=history_size)
        
        # Thresholds para ángulos en grados
        self.thresholds = {
            'knee_angle_min': 140,      # Ángulo mínimo de rodilla en posición inicial
            'knee_angle_max': 180,      # Ángulo máximo de rodilla
            'hip_angle_min': 45,        # Ángulo mínimo de cadera en fase down
            'hip_angle_max': 170,       # Ángulo máximo de cadera en posición inicial
            'back_straightness': 15,    # Desviación máxima permitida en espalda
            'shoulder_hip_alignment': 0.1,  # Alineación hombro-cadera
            'bar_path_deviation': 0.15   # Desviación máxima de la barra
        }
    
    def calculate_angle(self, p1, p2, p3):
        """
        Calcula el ángulo entre tres puntos
        
        Args:
            p1, p2, p3: Diccionarios con coordenadas x, y
            
        Returns:
            Ángulo en grados
        """
        try:
            # Vectores
            v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
            v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
            
            # Calcular ángulo
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Evitar errores de precisión
            angle = np.arccos(cos_angle)
            
            return math.degrees(angle)
        except:
            return 0
    
    def calculate_distance(self, p1, p2):
        """Calcula distancia euclidiana entre dos puntos"""
        return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
    
    def check_knee_angle(self, landmarks):
        """Verifica ángulos de rodillas"""
        errors = []
        
        # Landmarks para piernas (MediaPipe pose landmarks)
        # 23: Cadera izq, 25: Rodilla izq, 27: Tobillo izq
        # 24: Cadera der, 26: Rodilla der, 28: Tobillo der
        
        try:
            # Pierna izquierda
            left_knee_angle = self.calculate_angle(
                landmarks[23], landmarks[25], landmarks[27]
            )
            
            # Pierna derecha  
            right_knee_angle = self.calculate_angle(
                landmarks[24], landmarks[26], landmarks[28]
            )
            
            # Verificar ángulos
            if left_knee_angle < self.thresholds['knee_angle_min']:
                errors.append({
                    'message': 'Rodilla izq muy flexionada',
                    'severity': 'critical',
                    'angle': left_knee_angle
                })
                
            if right_knee_angle < self.thresholds['knee_angle_min']:
                errors.append({
                    'message': 'Rodilla der muy flexionada', 
                    'severity': 'critical',
                    'angle': right_knee_angle
                })
                
            # Verificar simetría
            angle_diff = abs(left_knee_angle - right_knee_angle)
            if angle_diff > 20:
                errors.append({
                    'message': 'Rodillas asimétricas',
                    'severity': 'warning',
                    'difference': angle_diff
                })
                
        except Exception as e:
            print(f"Error calculando ángulos de rodilla: {e}")
            
        return errors
    
    def check_hip_angle(self, landmarks):
        """Verifica ángulo de cadera"""
        errors = []
        
        try:
            # 11: Hombro izq, 23: Cadera izq, 25: Rodilla izq
            left_hip_angle = self.calculate_angle(
                landmarks[11], landmarks[23], landmarks[25]
            )
            
            # 12: Hombro der, 24: Cadera der, 26: Rodilla der
            right_hip_angle = self.calculate_angle(
                landmarks[12], landmarks[24], landmarks[26]
            )
            
            avg_hip_angle = (left_hip_angle + right_hip_angle) / 2
            
            # Verificar si la cadera está demasiado alta o baja
            if avg_hip_angle < self.thresholds['hip_angle_min']:
                errors.append({
                    'message': 'Cadera muy baja',
                    'severity': 'warning',
                    'angle': avg_hip_angle
                })
            elif avg_hip_angle > self.thresholds['hip_angle_max']:
                errors.append({
                    'message': 'Cadera muy alta',
                    'severity': 'warning',
                    'angle': avg_hip_angle
                })
                
        except Exception as e:
            print(f"Error calculando ángulo de cadera: {e}")
            
        return errors
    
    def check_back_straightness(self, landmarks):
        """Verifica rectitud de la espalda"""
        errors = []
        
        try:
            # Puntos de la columna: hombros y caderas
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            # Calcular punto medio de hombros y caderas
            shoulder_mid = {
                'x': (left_shoulder['x'] + right_shoulder['x']) / 2,
                'y': (left_shoulder['y'] + right_shoulder['y']) / 2
            }
            
            hip_mid = {
                'x': (left_hip['x'] + right_hip['x']) / 2,
                'y': (left_hip['y'] + right_hip['y']) / 2
            }
            
            # Calcular ángulo de inclinación de la espalda
            spine_vector = np.array([shoulder_mid['x'] - hip_mid['x'], 
                                   shoulder_mid['y'] - hip_mid['y']])
            vertical_vector = np.array([0, 1])
            
            cos_angle = np.dot(spine_vector, vertical_vector) / np.linalg.norm(spine_vector)
            spine_angle = math.degrees(math.acos(abs(cos_angle)))
            
            # Verificar si la espalda está muy curvada
            if spine_angle > 45:  # Más de 45 grados de inclinación
                errors.append({
                    'message': 'Espalda muy inclinada',
                    'severity': 'critical',
                    'angle': spine_angle
                })
                
        except Exception as e:
            print(f"Error verificando espalda: {e}")
            
        return errors
    
    def check_shoulder_position(self, landmarks):
        """Verifica posición de hombros"""
        errors = []
        
        try:
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            # Verificar si los hombros están por delante de las caderas
            shoulder_mid_x = (left_shoulder['x'] + right_shoulder['x']) / 2
            hip_mid_x = (left_hip['x'] + right_hip['x']) / 2
            
            forward_lean = shoulder_mid_x - hip_mid_x
            
            if abs(forward_lean) > self.thresholds['shoulder_hip_alignment']:
                if forward_lean > 0:
                    errors.append({
                        'message': 'Hombros muy adelante',
                        'severity': 'warning',
                        'deviation': forward_lean
                    })
                else:
                    errors.append({
                        'message': 'Hombros muy atrás',
                        'severity': 'warning', 
                        'deviation': abs(forward_lean)
                    })
                    
        except Exception as e:
            print(f"Error verificando hombros: {e}")
            
        return errors
    
    def check_foot_position(self, landmarks):
        """Verifica posición de los pies"""
        errors = []
        
        try:
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]
            left_heel = landmarks[29]
            right_heel = landmarks[30]
            
            # Verificar ancho de postura
            foot_width = abs(left_ankle['x'] - right_ankle['x'])
            
            if foot_width < 0.1:  # Pies muy juntos
                errors.append({
                    'message': 'Pies muy juntos',
                    'severity': 'warning',
                    'width': foot_width
                })
            elif foot_width > 0.4:  # Pies muy separados
                errors.append({
                    'message': 'Pies muy separados',
                    'severity': 'warning',
                    'width': foot_width
                })
                
        except Exception as e:
            print(f"Error verificando pies: {e}")
            
        return errors
    
    def check_deadlift_form(self, landmarks, current_stage):
        """
        Función principal que verifica todos los aspectos de la forma
        
        Args:
            landmarks: Diccionario con landmarks de MediaPipe
            current_stage: Etapa actual del ejercicio ('up', 'down', etc.)
            
        Returns:
            Lista de errores detectados
        """
        all_errors = []
        
        if landmarks and len(landmarks) >= 33:  # MediaPipe tiene 33 landmarks
            # Verificar diferentes aspectos
            all_errors.extend(self.check_knee_angle(landmarks))
            all_errors.extend(self.check_hip_angle(landmarks))
            all_errors.extend(self.check_back_straightness(landmarks))
            all_errors.extend(self.check_shoulder_position(landmarks))
            all_errors.extend(self.check_foot_position(landmarks))
            
            # Almacenar en historial
            self.error_history.append(all_errors)
            
            # Filtrar errores consistentes (que aparecen en múltiples frames)
            consistent_errors = self._get_consistent_errors()
            
            return consistent_errors
        
        return []
    
    def _get_consistent_errors(self):
        """Obtiene errores que aparecen consistentemente en el historial"""
        if len(self.error_history) < 3:
            return list(self.error_history[-1]) if self.error_history else []
        
        # Contar frecuencia de cada tipo de error
        error_counts = {}
        recent_errors = list(self.error_history)[-5:]  # Últimos 5 frames
        
        for frame_errors in recent_errors:
            for error in frame_errors:
                key = error['message']
                if key not in error_counts:
                    error_counts[key] = {'count': 0, 'error': error}
                error_counts[key]['count'] += 1
        
        # Retornar errores que aparecen en al menos 3 de los últimos 5 frames
        consistent_errors = []
        for key, data in error_counts.items():
            if data['count'] >= 3:
                consistent_errors.append(data['error'])
        
        return consistent_errors
    
    def reset_error_history(self):
        """Resetea el historial de errores"""
        self.error_history.clear()
        self.angle_history.clear()