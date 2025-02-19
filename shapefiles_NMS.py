from ultralytics import YOLO
import geopandas as gpd
from shapely.geometry import Polygon
import cv2
import numpy as np

# Cargar el modelo
model = YOLO("modelos/MODELO_RUNPOD.pt")

# Realizar la predicción
results = model.predict(source="testimages/imagen.tif", show=True, save=True, conf=0.2)

# Listas para almacenar las bounding boxes, puntuaciones y clases
boxes = []
scores = []
class_ids = []

# Procesar los resultados de la predicción
for result in results:
    for box in result.boxes:
        # Obtener las coordenadas de la bounding box
        x_center, y_center, width, height = box.xywh[0].tolist()
        
        # Convertir de (x_center, y_center, width, height) a (x_min, y_min, x_max, y_max)
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2
        
        # Agregar a las listas
        boxes.append([x_min, y_min, x_max, y_max])
        scores.append(box.conf.item())  # Confianza de la detección
        class_ids.append(box.cls.item())  # ID de la clase

# Convertir a numpy arrays
boxes = np.array(boxes)
scores = np.array(scores)
class_ids = np.array(class_ids)

# Aplicar Non-Maximum Suppression (NMS)
# Parámetros:
#   boxes: las cajas delimitadoras
#   scores: las confianzas de las detecciones
#   score_threshold: umbral mínimo de confianza para considerar una detección
#   nms_threshold: umbral de IoU para considerar que dos cajas se superponen
score_threshold = 0.2  # Mismo valor que model.predict
nms_threshold = 0.65 # Ajusta este valor según tus necesidades

# Aplicar NMS
indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold)

# Filtrar las detecciones usando los índices de NMS
filtered_boxes = boxes[indices]
filtered_scores = scores[indices]
filtered_class_ids = class_ids[indices]

# Lista para almacenar los polígonos de las bounding boxes filtradas
polygons = []

# Crear polígonos a partir de las bounding boxes filtradas
for box in filtered_boxes:
    x_min, y_min, x_max, y_max = box
    polygon = Polygon([
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max)
    ])
    polygons.append(polygon)

# Crear un GeoDataFrame con los polígonos
gdf = gpd.GeoDataFrame(geometry=polygons)

# Asignar un sistema de referencia de coordenadas local (píxeles)
gdf.set_crs(epsg=3857, inplace=True)  # Puedes usar un CRS local o uno genérico como EPSG:3857

# Guardar el GeoDataFrame como un shapefile
gdf.to_file("shapefiles/detectionsX5.shp")

print("Shapefile guardado como 'detectionsX.shp'")