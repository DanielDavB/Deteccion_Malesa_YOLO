from ultralytics import YOLO
import cv2

# Cargar el modelo
model = YOLO("MODELO_RUNPOD.pt")

# Realizar la predicción
results = model.predict(source="imagen.tif", show=False, save=True, conf=0.2)
