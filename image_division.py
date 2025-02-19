import rasterio
from rasterio.windows import Window
import os
import math

def calculate_grid_size(file_size_mb):
    """
    Calcula el tamaño de la cuadrícula en función del tamaño del archivo TIFF.
    """
    base_grid = (5, 5)
    if file_size_mb > 500:
        factor = math.ceil(math.sqrt(file_size_mb / 500))
        return (base_grid[0] * factor, base_grid[1] * factor)
    return base_grid

def split_tif_to_tif(tif_path, output_dir):
    """
    Divide un archivo TIFF en una cuadrícula ajustada al tamaño del archivo y guarda las secciones resultantes como archivos TIFF.

    Args:
        tif_path (str): Ruta del archivo TIFF de entrada.
        output_dir (str): Carpeta donde se guardarán los archivos TIFF.
    """
    # Crear la carpeta de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Obtener el tamaño del archivo en MB
    file_size_mb = os.path.getsize(tif_path) / (1024 * 1024)
    grid_size = calculate_grid_size(file_size_mb)

    # Abrir el archivo TIFF
    with rasterio.open(tif_path) as src:
        print(f"El archivo TIFF tiene {src.count} bandas (canales). Tamaño: {file_size_mb:.2f} MB. Dividiendo en {grid_size} partes.")
        width = src.width
        height = src.height

        tile_width = math.ceil(width / grid_size[1])
        tile_height = math.ceil(height / grid_size[0])

        for row in range(grid_size[0]):
            for col in range(grid_size[1]):
                # Definir los límites de la ventana
                col_off = col * tile_width
                row_off = row * tile_height
                w = min(tile_width, width - col_off)
                h = min(tile_height, height - row_off)
                
                if w <= 0 or h <= 0:
                    continue  # Evita ventanas fuera de los límites
                
                window = Window(col_off=col_off, row_off=row_off, width=w, height=h)
                
                # Leer los datos de la ventana
                data = src.read(window=window)
                
                # Definir los metadatos para el nuevo archivo TIFF
                profile = src.profile.copy()
                profile.update({
                    "width": w,
                    "height": h,
                    "transform": rasterio.windows.transform(window, src.transform)
                })

                # Guardar la sección como un nuevo archivo TIFF
                output_file = os.path.join(output_dir, f"tile_{row}_{col}.tif")
                with rasterio.open(output_file, "w", **profile) as dst:
                    dst.write(data)

                print(f"Sección guardada: {output_file}")

if __name__ == "__main__":
    tif_path = r"C:\Users\danie\OneDrive\Documentos\Deteccion_Malesa_AI\testimages\bigimage.tif"  # Reemplaza con la ruta de tu archivo TIFF
    output_dir = r"C:\Users\danie\OneDrive\Documentos\Deteccion_Malesa_AI\testimages"  # Reemplaza con la carpeta de salida
    split_tif_to_tif(tif_path, output_dir)
