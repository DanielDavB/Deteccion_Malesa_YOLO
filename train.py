# from ultralytics import YOLO

# model = YOLO("yolo11n-seg.pt") #Modelo que descargamos
# model.train(data="dataset_custom.yaml", imgsz=640, epochs=100, workers= 0, device = 0)  # train

if __name__ == '__main__':
    from ultralytics import YOLO  # O la librería que uses
    
    model = YOLO('yolo11n-seg.pt')  # Ajusta según tu modelo
    results = model.train(data="dataset_custom.yaml", epochs=150, imgsz=640, workers=0)
