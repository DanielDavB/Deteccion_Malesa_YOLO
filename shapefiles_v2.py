from ultralytics import YOLO
import geopandas as gpd
from shapely.geometry import Polygon
import cv2
import numpy as np
from shapely.ops import unary_union

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
        x_center, y_center, width, height = box.xywh[0].tolist()
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2
        
        boxes.append([x_min, y_min, x_max, y_max])
        scores.append(box.conf.item())
        class_ids.append(box.cls.item())

boxes = np.array(boxes)
scores = np.array(scores)
class_ids = np.array(class_ids)

# Crear polígonos a partir de las bounding boxes
polygons = [Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]) for x_min, y_min, x_max, y_max in boxes]

# Filtrar bounding boxes eliminando aquellas que se solapan excesivamente
filtered_polygons = []
for poly in polygons:
    if not any(poly.intersects(other) and poly != other for other in filtered_polygons):
        filtered_polygons.append(poly)

# Fusionar bounding boxes superpuestas automáticamente
merged_polygons = unary_union(filtered_polygons)

# Convertir el resultado a una lista de polígonos si es necesario
if merged_polygons.geom_type == "Polygon":
    merged_polygons = [merged_polygons]
else:
    merged_polygons = list(merged_polygons.geoms)

# Crear un GeoDataFrame con los polígonos fusionados
gdf = gpd.GeoDataFrame(geometry=merged_polygons)
gdf.set_crs(epsg=3857, inplace=True)  # Puedes cambiar el CRS según tu caso

# Guardar el GeoDataFrame como un shapefile
gdf.to_file("shapefiles/detections_merged2.shp")

print("Shapefile guardado como 'detections_merged.shp'")
