import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from landmarks import landmarks

# Cargar datos
df = pd.read_csv('squats_data.csv')
X = df[landmarks]
y = df['class']

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo de squats: {accuracy:.2f}")

# Guardar modelo
with open('squats.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Modelo guardado como 'squats.pkl'")