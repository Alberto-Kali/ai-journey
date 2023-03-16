import argparse
import imutils
import time
import cv2
from imutils.video import VideoStream
from urllib.request import urlopen
import numpy as np
import sys


host = 'http://192.168.0.101:8080/'
url = host + 'image.jpg'

ap = argparse.ArgumentParser()
ap.add_argument("-p", required=True,
	help="prototxt файл")
ap.add_argument("-m", required=True,
	help="файл модели")
ap.add_argument("-s", required=True, 
	help="Источник видео (webcam/host)")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="Минимальная вероятность фильтрации слабых обнаружений")
args = vars(ap.parse_args())


Cl = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
Colors = np.random.uniform(0, 255, size=(len(Cl), 3))

# Загрузить модель
print("[И] Загружаю модель...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Инициализировать камеру
print("[И] Запускаю стрим...")

if args["s"] == "webcam":
	vs = cv2.VideoCapture(0)

time.sleep(2.0)

detected_objects = []
# Зациклить получение кадров
while True:
	# Маштабировать кадр
	if args["s"] == "webcam":
		ret, frame = vs.read()
	else:
		imgResp=urlopen(url)
		imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
		frame=cv2.imdecode(imgNp,-1)
	
	frame = imutils.resize(frame, width=800)
		
	# Взять размеры рамки и преобразовать ее в большой объект
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# Пропустить большой двоичный объект через сеть и получить обнаружения и прогнозы
	net.setInput(blob)
	detections = net.forward()
    
	# Зациклить распознавание
	for i in np.arange(0, detections.shape[2]):
		# Извлечь достоверность связанную с предсказанием
		confidence = detections[0, 0, i, 2]

		# Отфильтровать слабые обнаружения, убедившись, что полученная достоверность превышает минимальную достоверность
		if confidence > args["confidence"]:
			# Извлечь индекс метки класса из `detections`, затем вычислить (x, y)-координаты ограничивающей рамки для объекта
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Нарисовать предсказания на кадре
			label = "{}: {:.2f}%".format(Cl[idx],
				confidence * 100)
			detected_objects.append(label)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				Colors[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors[idx], 2)
			
	
	# Показать финальный кадр
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# Q чтобы выйдти
	if key == ord("q"):
		break


# Разбить все окна
cv2.destroyAllWindows()
