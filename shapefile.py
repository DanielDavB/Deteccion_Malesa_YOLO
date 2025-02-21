import rasterio
from rasterio.windows import Window
from rasterio.merge import merge
import os
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
from shapely.ops import unary_union
import pandas as pd

def calculate_grid_size(file_size_mb):
    """
    Calcula el tamaño de la cuadrícula en función del tamaño del archivo TIFF.
    """
    base_grid = (5, 5)
    if file_size_mb > 500:
        factor = math.ceil(math.sqrt(file_size_mb / 500))
        return (base_grid[0] * factor, base_grid[1] * factor)
    return base_grid

def analyze_image(image_path):
    """
    Analiza una imagen usando YOLO y guarda el resultado como un shapefile.
    """
    model = YOLO("modelos/MODELO_RUNPOD.pt")
    results = model.predict(source=image_path, show=False, save=True, conf=0.2)
    
    boxes = []
    for result in results:
        for box in result.boxes:
            x_center, y_center, width, height = box.xywh[0].tolist()
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            
            boxes.append([x_min, y_min, x_max, y_max])
    
    boxes = np.array(boxes)
    polygons = [Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]) for x_min, y_min, x_max, y_max in boxes]
    merged_polygons = unary_union(polygons)
    if merged_polygons.geom_type == "Polygon":
        merged_polygons = [merged_polygons]
    else:
        merged_polygons = list(merged_polygons.geoms)

    gdf = gpd.GeoDataFrame(geometry=merged_polygons)
    gdf.set_crs(epsg=3857, inplace=True)
    output_shapefile = image_path.replace(".tif", "_detections.shp")
    gdf.to_file(output_shapefile)
    print(f"Shapefile guardado: {output_shapefile}")

def merge_tifs(output_dir, merged_tif_path):
    """
    Une todas las imágenes TIFF en una sola.
    """
    tif_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".tif")]
    src_files_to_mosaic = [rasterio.open(f) for f in tif_files]
    mosaic, out_trans = merge(src_files_to_mosaic)
    
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_trans})
    
    with rasterio.open(merged_tif_path, "w", **out_meta) as dest:
        dest.write(mosaic)
    print(f"Imagen TIFF fusionada guardada en: {merged_tif_path}")

def merge_shapefiles(output_dir, merged_shapefile_path):
    """
    Fusiona todos los shapefiles en un único archivo.
    """
    shp_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith("_detections.shp")]
    gdfs = [gpd.read_file(f) for f in shp_files]
    merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    merged_gdf.to_file(merged_shapefile_path)
    print(f"Shapefile fusionado guardado en: {merged_shapefile_path}")

def split_tif_to_tif(tif_path, output_dir, num_threads=4):
    """
    Divide un archivo TIFF en una cuadrícula ajustada al tamaño del archivo y guarda las secciones resultantes como archivos TIFF.
    Luego, analiza cada sección en paralelo usando un número limitado de hilos.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_size_mb = os.path.getsize(tif_path) / (1024 * 1024)
    grid_size = calculate_grid_size(file_size_mb)
    
    with rasterio.open(tif_path) as src:
        print(f"El archivo TIFF tiene {src.count} bandas (canales). Tamaño: {file_size_mb:.2f} MB. Dividiendo en {grid_size} partes.")
        width, height = src.width, src.height
        tile_width = math.ceil(width / grid_size[1])
        tile_height = math.ceil(height / grid_size[0])
        
        image_paths = []
        
        for row in range(grid_size[0]):
            for col in range(grid_size[1]):
                col_off = col * tile_width
                row_off = row * tile_height
                w = min(tile_width, width - col_off)
                h = min(tile_height, height - row_off)
                
                if w <= 0 or h <= 0:
                    continue
                
                window = Window(col_off=col_off, row_off=row_off, width=w, height=h)
                data = src.read(window=window)
                profile = src.profile.copy()
                profile.update({"width": w, "height": h, "transform": rasterio.windows.transform(window, src.transform)})
                
                output_file = os.path.join(output_dir, f"tile_{row}_{col}.tif")
                with rasterio.open(output_file, "w", **profile) as dst:
                    dst.write(data)
                
                print(f"Sección guardada: {output_file}")
                image_paths.append(output_file)
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            executor.map(analyze_image, image_paths)
    
    merge_tifs(output_dir, os.path.join(output_dir, "merged_output.tif"))
    merge_shapefiles(output_dir, os.path.join(output_dir, "merged_detections.shp"))

if __name__ == "__main__":
    tif_path = r"C:\Users\danie\OneDrive\Documentos\Deteccion_Malesa_AI\testimages\imagen.tif"  # Reemplaza con la ruta de tu archivo TIFF
    output_dir = r"C:\Users\danie\OneDrive\Documentos\Deteccion_Malesa_AI\testimages"  # Reemplaza con la carpeta de salida
    num_threads = 4  # Ajusta el número de hilos según tu CPU
    split_tif_to_tif(tif_path, output_dir, num_threads)
