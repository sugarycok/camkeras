import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
import cv2
from keras.models import load_model
from PIL import ImageOps
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("untitled.ui", self)  # 프로젝트 폴더에 있는 UI 파일을 로드

        # Load the model
        self.model = load_model("keras_Model.h5", compile=False)

        # Load the labels
        self.class_names = open("labels.txt", "r", encoding="utf-8").readlines()

        # Open the webcam
        self.cap = cv2.VideoCapture(0)

        # Start timer to update UI
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateUI)
        self.timer.start(100)  # 100ms 마다 UI 업데이트

        # Connect FinishButton click event to finishApp method
        self.FinishButton.clicked.connect(self.finishApp)

    def finishApp(self):
        # 종료할지 묻는 메시지 박스 표시
        reply = QMessageBox.question(self, '종료', '프로그램을 종료하시겠습니까?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            # 프로그램 종료
            QApplication.quit()

    def updateUI(self):
        ret, frame = self.cap.read()
        if ret:
            # Display frame in label
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytesPerLine = ch * w
            qImg = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            self.label.setPixmap(pixmap)
            
            # Preprocess the frame
            processed_frame = self.preprocess_image(frame)

            # Expand dimensions to match the model's input requirements
            data = np.expand_dims(processed_frame, axis=0)

            # Predict the class
            prediction = self.model.predict(data)

            # Update progress bars for each class
            for i in range(len(prediction[0])):
                class_name = self.class_names[i]
                confidence_score = prediction[0][i]
                progress_bar = getattr(self, f"progressBar_{i+1}")
                progress_bar.setValue(int(confidence_score * 100))  # 퍼센트로 표시

    def preprocess_image(self, image):
        # Resize the image to 224x224 and then crop from the center
        size = (224, 224)
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        # Convert image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Normalize the image
        normalized_image_array = (image.astype(np.float32) / 127.5) - 1
        return normalized_image_array

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
