import cv2
import torch
from ultralytics import YOLO
import pytesseract
from PIL import Image


pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Harsh kumar\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


model = YOLO('best.pt')  
image_path = 'car.jpg' 
image = cv2.imread(image_path)
results = model(image)
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box[:4]
        if score > 0.5 and cls == 0:
            plate_img = image[int(y1):int(y2), int(x1):int(x2)]
            plate_pil = Image.fromarray(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
            plate_text = pytesseract.image_to_string(plate_pil, config='--psm 8')
            print(f'Detected Number Plate: {plate_text}')



# Diplaying image detections
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box[:4]
        if score > 0.5 and cls == 0:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

cv2.imshow('Detected Plates', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
