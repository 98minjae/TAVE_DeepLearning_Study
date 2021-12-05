from objection_detect_annotate import DetectionObj
from threading import Thread
import cv2

def resize(image, new_width=None, new_height=None):
    """
    이미지 크기를 새로운 너비나 높이를 기준으로 원래 가로/세로 비율에 맞춰 조정
    """
    height, width, depth = image.shape
    if new_width:
        new_height = int((new_width / float(width)) * height)
    elif new_height:
        new_width = int((new_height / float(height)) * width)
    else:
        return image
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

class WebcamStream:
    def __init__(self):
        # 웹캠 초기화
        self.stream = cv2.VideoCapture(0)
        # SSD Mobilenet으로 텐서플로 API 시작
        self.detection = DetectionObj(model='ssd_mobilenet_v1_coco_11_06_2017')
        # 웹캠이 자체 튜닝하도록 동영상 캡처 시작
        _, self.frame = self.stream.read()
        # stop 플래그에 False 설정
        self.stop = False
        #
        Thread(target=self.refresh, args=()).start()

    def refresh(self):
        # 함수 외부에서 명시적인 중지 명령이 올 떄까지 반복
        while True:
            if self.stop:
                return
            _, self.frame = self.stream.read()

    def get(self):
        # 주석이 달린 이미지 반환
        return self.detection.annotate_photogram(self.frame)

    def halt(self):
        # 중지 플래그 설정
        self.stop = True

if __name__ == "__main__":
    stream = WebcamStream()

    while True:
        # 스레드에 들어오는 동영상 스트림에서 프레임을 잡아
        # 최대 너비가 400 픽셀이 되도록 크기를 조정
        frame = resize(stream.get(), new_width=400)
        cv2.imshow("Webcam", frame)
        # 스페이스 바를 누르면 프로그램이 중지함
        if cv2.waitKey(1) & 0xFF == ord(" "):
            # 먼저 동영상 스트리밍 스레드가 중지
            stream.halt()
            # 그다음 while 루프 중지지
            break