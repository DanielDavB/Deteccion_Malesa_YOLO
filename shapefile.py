from ultralytics import YOLO
import geopandas as gpd
from shapely.geometry import Polygon

# Cargar el modelo
model = YOLO("modelos/MODELO_RUNPOD.pt")

# Realizar la predicción
results = model.predict(source="imagen.tif", show=True, save=True, conf=0.2)

# Lista para almacenar los polígonos de las bounding boxes
polygons = []

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
        
        # Crear un polígono con las coordenadas de píxeles
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
gdf.to_file("detections.shp")

print("Shapefile guardado como 'detections.shp'")