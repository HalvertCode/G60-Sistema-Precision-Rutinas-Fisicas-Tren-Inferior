import numpy as np
import math

class LungesFormChecker:
    def __init__(self):
        self.error_history = []
        self.frame_count = 0
        self.error_threshold = 5  # Número de frames consecutivos para confirmar error
        
    def calculate_angle(self, a, b, c):
        """Calcula el ángulo entre tres puntos"""
        a = np.array(a)
        b = np.array(b)  # vértice
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle
    
    def calculate_distance(self, point1, point2):
        """Calcula la distancia euclidiana entre dos puntos"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def check_knee_alignment(self, hip, knee, ankle):
        """Verifica si la rodilla está correctamente alineada"""
        # Calcular el ángulo de la rodilla
        knee_angle = self.calculate_angle(hip, knee, ankle)
        
        errors = []
        
        # Verificar ángulo de rodilla crítico (demasiado cerrado)
        if knee_angle < 70:
            errors.append({
                'type': 'knee_too_bent',
                'severity': 'critical',
                'message': f'Rodilla muy flexionada ({knee_angle:.1f}°)'
            })
        
        # Verificar ángulo de rodilla incorrecto (muy abierto)
        elif knee_angle > 100:
            errors.append({
                'type': 'knee_too_straight',
                'severity': 'moderate',
                'message': f'Rodilla poco flexionada ({knee_angle:.1f}°)'
            })
        
        return errors, knee_angle
    
    def check_back_posture(self, shoulder_left, shoulder_right, hip_left, hip_right):
        """Verifica la postura de la espalda"""
        errors = []
        
        # Calcular el centro de hombros y caderas
        shoulder_center = [(shoulder_left[0] + shoulder_right[0])/2, 
                          (shoulder_left[1] + shoulder_right[1])/2]
        hip_center = [(hip_left[0] + hip_right[0])/2, 
                     (hip_left[1] + hip_right[1])/2]
        
        # Calcular inclinación del torso
        torso_angle = math.atan2(shoulder_center[0] - hip_center[0], 
                                hip_center[1] - shoulder_center[1]) * 180 / math.pi
        
        # Verificar inclinación excesiva hacia adelante
        if abs(torso_angle) > 20:
            severity = 'critical' if abs(torso_angle) > 35 else 'moderate'
            errors.append({
                'type': 'torso_lean',
                'severity': severity,
                'message': f'Torso inclinado {abs(torso_angle):.1f}°'
            })
        
        return errors
    
    def check_foot_position(self, knee_front, ankle_front, knee_back, ankle_back):
        """Verifica la posición de los pies"""
        errors = []
        
        # Verificar que el pie delantero esté correctamente posicionado
        # El tobillo debe estar aproximadamente debajo de la rodilla
        knee_ankle_distance_front = abs(knee_front[0] - ankle_front[0])
        
        if knee_ankle_distance_front > 0.1:  # Umbral ajustable
            errors.append({
                'type': 'front_foot_position',
                'severity': 'moderate',
                'message': 'Pie delantero muy adelante'
            })
        
        # Verificar posición del pie trasero
        if ankle_back[1] < knee_back[1]:  # Pie trasero más alto que rodilla
            errors.append({
                'type': 'back_foot_position',
                'severity': 'critical',
                'message': 'Pie trasero mal posicionado'
            })
        
        return errors
    
    def check_knee_safety(self, knee, ankle):
        """Verifica que la rodilla no sobrepase el tobillo (safety check)"""
        errors = []
        
        # La rodilla no debe estar muy por delante del tobillo
        if knee[0] > ankle[0] + 0.05:  # Margen de seguridad
            errors.append({
                'type': 'knee_over_toe',
                'severity': 'critical',
                'message': 'Rodilla sobrepasa el pie'
            })
        
        return errors
    
    def determine_front_back_legs(self, knee_left, knee_right, ankle_left, ankle_right):
        """Determina cuál pierna está adelante y cuál atrás"""
        # La pierna delantera generalmente tiene el tobillo más adelante (menor Y)
        if ankle_left[1] < ankle_right[1]:
            return (knee_left, ankle_left), (knee_right, ankle_right), 'left_front'
        else:
            return (knee_right, ankle_right), (knee_left, ankle_left), 'right_front'
    
    def analyze_form(self, landmarks):
        """Analiza la forma general de la zancada"""
        self.frame_count += 1
        current_errors = []
        
        try:
            # Extraer landmarks
            hip_left = landmarks['hip_left']
            knee_left = landmarks['knee_left']
            ankle_left = landmarks['ankle_left']
            hip_right = landmarks['hip_right']
            knee_right = landmarks['knee_right']
            ankle_right = landmarks['ankle_right']
            shoulder_left = landmarks['shoulder_left']
            shoulder_right = landmarks['shoulder_right']
            
            # Verificar alineación de rodillas
            left_errors, left_angle = self.check_knee_alignment(hip_left, knee_left, ankle_left)
            right_errors, right_angle = self.check_knee_alignment(hip_right, knee_right, ankle_right)
            
            current_errors.extend(left_errors)
            current_errors.extend(right_errors)
            
            # Verificar postura de espalda
            back_errors = self.check_back_posture(shoulder_left, shoulder_right, hip_left, hip_right)
            current_errors.extend(back_errors)
            
            # Determinar pierna delantera y trasera
            (knee_front, ankle_front), (knee_back, ankle_back), front_leg = self.determine_front_back_legs(
                knee_left, knee_right, ankle_left, ankle_right
            )
            
            # Verificar posición de pies
            foot_errors = self.check_foot_position(knee_front, ankle_front, knee_back, ankle_back)
            current_errors.extend(foot_errors)
            
            # Verificar seguridad de rodillas
            safety_errors_front = self.check_knee_safety(knee_front, ankle_front)
            safety_errors_back = self.check_knee_safety(knee_back, ankle_back)
            
            current_errors.extend(safety_errors_front)
            current_errors.extend(safety_errors_back)
            
            # Verificar simetría de la zancada
            if abs(left_angle - right_angle) > 15:
                current_errors.append({
                    'type': 'asymmetry',
                    'severity': 'moderate',
                    'message': f'Zancada asimétrica ({abs(left_angle - right_angle):.1f}°)'
                })
            
            # Verificar profundidad de la zancada
            if left_angle > 95 and right_angle > 95:
                current_errors.append({
                    'type': 'shallow_lunge',
                    'severity': 'moderate',
                    'message': 'Zancada muy superficial'
                })
            
            # Filtrar errores persistentes
            persistent_errors = self.filter_persistent_errors(current_errors)
            
            return persistent_errors
            
        except Exception as e:
            print(f"Error en análisis de forma: {e}")
            return []
    
    def filter_persistent_errors(self, current_errors):
        """Filtra errores que persisten por varios frames"""
        # Agregar errores actuales al historial
        self.error_history.append(current_errors)
        
        # Mantener solo los últimos frames
        if len(self.error_history) > self.error_threshold:
            self.error_history.pop(0)
        
        # Contar frecuencia de cada tipo de error
        error_counts = {}
        for frame_errors in self.error_history:
            for error in frame_errors:
                error_type = error['type']
                if error_type not in error_counts:
                    error_counts[error_type] = {'count': 0, 'error': error}
                error_counts[error_type]['count'] += 1
        
        # Retornar errores que aparecen en al menos 3 de los últimos 5 frames
        persistent_errors = []
        min_count = max(1, len(self.error_history) // 2)
        
        for error_type, data in error_counts.items():
            if data['count'] >= min_count:
                persistent_errors.append(data['error'])
        
        return persistent_errors
    
    def clear_errors(self):
        """Limpia el historial de errores"""
        self.error_history = []
        self.frame_count = 0