import os
import numpy as np
import tensorflow as tf
import six.moves.urllib as urllib
import tarfile
from PIL import Image
from tqdm import tqdm
from time import gmtime, strftime
import json
import cv2

try:
  from moviepy.editor import VideoFileClip
except:
  # FFmpeg (https://www.ffmpeg.org)이 컴퓨터에 없으면
  # 인터넷에서 내려받음(인터넷 연결 필요)
  import imageio
  imageio.plugins.ffmpeg.download()
  from moviepy.editor import VideoFileClip

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class DetectionObj(object):
    """
    DetectionObj는 다양한 원천(파일, 웹캠에서 가져온 이미지, 동영상)에서
    가져온 이미지에 주석을 달기 위해 구글 텐서플로 detection API를 활용하기 적합한 클래스다.
    """

    def __init__(self, model='ssd_mobilenet_v1_coco_11_06_2017'):
        """
        클래스가 인스턴스화될 때 실행될 명령
        """
        # 파이썬 스크립트가 실행될 경로
        self.CURRENT_PATH = os.getcwd()

        # 주석을 저장할 경로 (수정 가능)
        self.TARGET_PATH = self.CURRENT_PATH

        # 텐서플로 모델 Zoo에서 미리 훈련된 탐지 모델 선택
        self.MODELS = [
            "ssd_mobilenet_v1_coco_11_06_2017"
        ]

        # 모델이 객체를 탐지할 때 사용할 임곗값 설정
        self.THRESHOLD = 0.25  # 실제로 가장 많이 사용하는 임곗값임

        # 선택한 미리 훈련된 탐지 모델이 사용 가능한지 확인
        if model in self.MODELS:
            self.MODEL_NAME = model
        else:
            # 사용할 수 없다면 기본 모델로 되돌림
            print("Model not available, reverted to default", self.MODELS[0])
            self.MODEL_NAME = self.MODELS[0]

        # 확정된 텐서플로 모델의 파일명
        self.CKPT_FILE = os.path.join(self.CURRENT_PATH, 'object_detection',
                                      self.MODEL_NAME, 'frozen_inference_graph.pb')

        # 탐지 모델 로딩
        # 디스크에 탐지 모델이 없다면, 인터넷에서 내려받음 (인터넷 연결 필요)
        try:
            self.DETECTION_GRAPH = self.load_frozen_model()
        except:
            print('Couldn\'t find', self.MODEL_NAME)
            self.download_frozen_model()
            self.DETECTION_GRAPH = self.load_frozen_model()

        # 탐지 모델에 의해 인식될 클래스 레이블 로딩
        self.NUM_CLASSES = 90
        path_to_labels = os.path.join(self.CURRENT_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')
        label_mapping = label_map_util.load_labelmap(path_to_labels)
        extracted_categories = label_map_util.convert_label_map_to_categories(label_mapping,
                                                                              max_num_classes=self.NUM_CLASSES,
                                                                              use_display_name=True)
        self.LABELS = {item['id']: item['name'] for item in extracted_categories}
        self.CATEGORY_INDEX = label_map_util.create_category_index(extracted_categories)
        # 범주 숙자 코드를 텍스트 표현과 연결한 딕셔너리를 포함하여 접근에 편의성을 제공하는 self.LABELS 변수를 갖게 됐다.

        # 텐서플로 세션 시작
        self.TF_SESSION = tf.compat.v1.Session(graph=self.DETECTION_GRAPH)

    def load_frozen_model(self):
        """
        ckpt 파일에 동결된 탐지 모델을 디스크에서 메모리로 적재
        """
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v2.io.gfile.GFile(self.CKPT_FILE, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        return detection_graph

    def download_frozen_model(self):
        """
        고정된 탐지 모델이 디스크에 없을 때 인터넷에서 내려받음
        """

        def my_hook(t):
            """
            URLopener를 모니터링하기 위해 tqdm 인스턴스를 감쌈
            """
            last_b = [0]

            def inner(b=1, bsize=1, tsize=None):
                if tsize is not None:
                    t.total = tsize
                t.update((b - last_b[0]) * bsize)
                last_b[0] = b

            return inner

        # 모델을 찾을 수 있는 url 열기
        model_filename = self.MODEL_NAME + '.tar.gz'
        download_url = 'http://download.tensorflow.org/models/object_detection'
        opener = urllib.request.URLopener()

        # tqdm 완료 추정을 사용해 모델 내려 받기
        print('Downloading ...')
        with tqdm() as t:
            opener.retrieve(download_url + model_filename, model_filename, reporthook=my_hook(t))

        # 내려 받은 tar 파일에서 모델 추출하기
        print('Extracting ...')
        tar_file = tarfile.open(model_filename)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.path.join(self.CURRENT_PATH, 'object_detection'))

    def load_image_from_disk(self, image_path):
        return Image.open(image_path)

    def load_image_into_numpy_array(self, image):
        """
        텐서플로 모델이 처리하기에 적합한 Numpy 배열로 변환하기 위해 필요
        """
        try:
            (im_width, im_height) = image.size
            return np.array(image.getdata()).reshape(
                (im_height, im_width, 3)).astype(np.uint8)
        except:
            # 이전 프로시저가 실패하면
            # 우리는 이미지가 이미 넘파이 ndarray라고 생각한다
            return image

    def detect(self, images, annotate_on_image=True):
        """
        이미지 리스트를 처리해서 탐지 모델에 제공하고
        모델로부터 이미지에 표시될 점수, 윤곽 상자, 예측 범주를 가져옴
        """
        if type(images) is not list:
            images = [images]
        results = list()
        for image in images:
            # 이미지를 배열 기반으로 나타내면
            # 상자와 상자 레이블을 가지고 결과 이미지를 준비하기 위해 나중에 나용될 것임
            image_np = self.load_image_into_numpy_array(image)

            # 모델은 [1, None, None, 3] 형상을 갖는 이미지를 기대하므로 차원을 확장함
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = self.DETECTION_GRAPH.get_tensor_by_name('image_tensor:0')

            # 각 상자는 이미지에서 특정 사물이 탐지된 부분을 나타냄
            boxes = self.DETECTION_GRAPH.get_tensor_by_name('detection_boxes:0')

            # 점수는 각 객체에 대한 신뢰 수준을 나타냄
            # 점수는 범주 레이블과 함께 결과 이미지에 나타낼 수 있음.
            scores = self.DETECTION_GRAPH.get_tensor_by_name('detection_scores:0')
            classes = self.DETECTION_GRAPH.get_tensor_by_name('detection_classes:0')
            num_detections = self.DETECTION_GRAPH.get_tensor_by_name('num_detections:0')

            # 여기서 실제로 객체가 탐지된
            (boxes, scores, classes, num_detections) = self.TF_SESSION.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            if annotate_on_image:
                new_image = self.detection_on_image(image_np, boxes, scores, classes)
                results.append((new_image, boxes, scores, classes, num_detections))
            else:
                results.append((image_np, boxes, scores, classes, num_detections))
            return results

    def detection_on_image(self, image_np, boxes, scores, classes):
        """
        이미지에 탐지된 범주로 탐지 상자 두기
        """
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.CATEGORY_INDEX,
            use_normalized_coordinates=True,
            line_thickness=8)
        return image_np

    # visualize_image 함수를 사용해 화면에 표시될 윤곽 상자가 추가된 새로운 이미지를 반환
    # (스크립트가 다른 이미지를 처리하러 지나가기 전 이미지가 화면에 머무는 시간에 해당하는 지연 매개변수 조정 가능)

    def visualize_image(self, image_np, image_size=(400, 300), latency=3, bluish_correction=True):
        # image_size: 화면에 표시될 이미의 원하는 크기
        # latency: 각 이미지가 화면에 표시될 시간을 초로 정의해서 다음 이미지로 넘어가기 전 객체 탐지 프로시저를 잠근다
        # bluish_correction: RGB -> RGR 보정
        height, width, depth = image_np.shape
        reshaper = height / float(image_size[0])
        width = int(width / reshaper)
        height = int(height / reshaper)
        id_img = 'preview_' + str(np.sum(image_np))
        cv2.startWindowThread()
        cv2.namedWindow(id_img, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(id_img, width, height)
        if bluish_correction:
            RGB_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            cv2.imshow(id_img, RGB_img)
        else:
            cv2.imshow(id_img, image_np)
        cv2.waitKey(latency * 1000)

    def serialize_annotations(self, boxes, scores, classes, filename='data.json'):
        # scores: 신뢰수준
        # classes: 탐지된 범주
        # boxes: 윤곽 상자의 꼭짓점을 이미지 높이와 너비의 비율로 표현
        """
        주석을 디스크 및 JSON 파일에 저장
        """
        threshold = self.THRESHOLD
        valid = [position for position, score in enumerate(scores[0]) if score > threshold]
        if len(valid) > 0:
            valid_scores = scores[0][valid].tolist()
            valid_boxes = boxes[0][valid].tolist()
            valid_class = [self.LABELS[int(a_class)] for a_class in classes[0][valid]]
            with open(filename, 'w') as outfile:
                json_data = json.dumps({'classes': valid_class, 'boxes': valid_boxes, 'scores': valid_scores})
                json.dump(json_data, outfile)

    def get_time(self):
        """
        실제 날짜와 시간을 보고하는 문자열 반환
        """
        return strftime("%Y-%m-%d_%Hh%Mm%Ss", gmtime())

    def annotate_photogram(self, photogram):
        """
        동영상에서 가져온 사진을 탐지 범주에 해당하는 윤곽 상자로 주석 달기
        """
        new_photogram, boxes, scores, classes, num_detections = self.detect(photogram)[0]
        return new_photogram

    def capture_webcam(self):
        """
        통합된 웹캠에서 이미지 캡처하기
        """

        def get_image(device):
            """
            카메라에서 단일 이미지를 캡처해서 PIL 형식으로 반환하는 내부 함수
            """
            retval, im = device.read()
            return im

        # 통합된 웹캠 설정하기
        camera_port = 0

        # 카메라가 주변 빛에 맞춰 조정하기 때문에 버려야 할 프레임 개수
        ramp_frames = 30

        # cv2.VideoCapture로 웹캠 초기화
        camera = cv2.VideoCapture(camera_port)

        # 카메라 램프 조절 - 카메라를 적절한 빍기 수준에 맞추기 때문에 이 프레임들은 모두 제거
        print("Setting the webcam")
        for i in range(ramp_frames):
            _ = get_image(camera)

        # 스냅샷 가져오기
        print("Now taking a snapshot ... ", end='')
        camera_capture = get_image(camera)
        print('Done')

        # 카메라를 해제하고 재활용할 수 있게 만듦
        del (camera)
        return camera_capture

    def file_pipeline(self, images, visualize=True):
        """
        디스크로부터 로딩할 이미지 리스트를 처리하고 주석을 달기 위한 파이프라인
        """
        if type(images) is not list:
            images = [images]
        for filename in images:
            single_image = self.load_image_from_disk(filename)
            for new_image, boxes, scores, classes, num_detections in self.detect(single_image):
                self.serialize_annotations(boxes, scores, classes, filename=filename + ".json")
                if visualize:
                    self.visualize_image(new_image)

    def video_pipeline(self, video, audio=False):
        """
        디스크 상의 동영상을 처리해서 윤곽 상자로 주석을 달기 위한 파이프라인
        결과는 주석이 추가된 새로운 동영상임
        """
        clip = VideoFileClip(video)
        new_video = video.split('/')
        new_video[-1] = "annotated_" + new_video[-1]
        new_video = '/'.join(new_video)
        print("Saving annotated video to", new_video)
        video_annotation = clip.fl_image(self.annotate_photogram)
        video_annotation.write_videofile(new_video, audio=audio)

    def webcam_pipeline(self):
        """
        내부 웹캠에서 얻은 이미지를 처리해서 주석을 달고 JSON 파일을 디스크에 저장하는 파이프라인
        """
        webcam_image = self.capture_webcam()
        filename = "webcam_" + self.get_time()
        saving_path = os.path.join(self.CURRENT_PATH, filename + ".jpg")
        cv2.imwrite(saving_path, webcam_image)
        new_image, boxes, scores, classes, num_detections = self.detect(webcam_image)[0]
        json_obj = {'classes': classes, 'boxes': boxes, 'scores': scores}
        self.serialize_annotations(boxes, scores, classes,
                                   filename=filename + ".json")
        self.visualize_image(new_image, bluish_correction=False)
