# 웹캠 영상 실시간 객체 인식 프로그램

## 설명
이 프로그램은 PyQt5를 사용하여 웹캠에서 영상을 실시간으로 캡처하고, 신경망 모델을 사용하여 영상에 대한 객체를 인식하는 간단한 응용 프로그램입니다.

## 사용된 기술

**PyQt5**: 사용자 인터페이스를 구현하기 위해 PyQt5를 사용하였습니다.<br>
**OpenCV**: 웹캠으로부터 영상을 캡처하기 위해 OpenCV를 사용하였습니다.<br>
**Keras**: 사전에 학습된 모델을 로드하여 영상에 대한 객체 인식을 수행하기 위해 Keras를 사용하였습니다.<br>
**Python**: 프로그램의 주요 언어로 Python을 사용하였습니다.<br>
****

## 프로그램 구조

**MainWindow 클래스**: PyQt5의 QMainWindow 클래스를 상속하며, 프로그램의 메인 창을 구현합니다.<br>
**updateUI 메서드**: 웹캠에서 프레임을 읽어와 UI를 업데이트하고, 신경망 모델을 사용하여 객체를 인식합니다.<br>
**finishApp 메서드**: 프로그램을 종료하기 전에 사용자에게 확인 메시지를 표시하고, 사용자가 확인을 선택하면 프로그램을 종료합니다.<br>
**preprocess_image 메서드**: 입력 이미지를 모델이 예측할 수 있는 형식으로 전처리합니다.<br>
****

## 사용 방법

keras_Model.h5와 labels.txt 파일을 준비합니다.<br>
웹캠을 연결하고, 프로그램을 실행합니다.<br>
웹캠에서 감지된 객체가 UI에 실시간으로 표시됩니다.<br>
프로그램을 종료하려면 'Finish' 버튼을 클릭합니다.<br>
****

## 요구 사항

Python 3<br>
PyQt5<br>
OpenCV<br>
Keras<br>
