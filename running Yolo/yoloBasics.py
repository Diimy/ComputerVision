from ultralytics import YOLO
import cv2

print("Running YOLOv8 object detection...\n")

# Carrega o modelo YOLOv8 e realiza a inferência na imagem
model = YOLO("yolov8n.pt")  
results= model("./running Yolo/images/car04.jpg", show=True) 

# Acessa a imagem com as anotações
annotated_frame = results[0].plot()

# Exibe com OpenCV manualmente
cv2.imshow("Detections", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()