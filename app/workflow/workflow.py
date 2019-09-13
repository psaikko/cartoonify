from __future__ import division
import png
import numpy as np
from pathlib import Path
import logging
from app.sketch import SketchGizeh
import subprocess
from csv import writer
import cv2
import time
import os

class Workflow(object):
    """controls execution of app
    """

    def __init__(self, dataset, imageprocessor):
        self._path = Path('')
        self._image_path = Path('')
        self._dataset = dataset
        self._image_processor = imageprocessor
        self._sketcher = None
        self._logger = logging.getLogger(self.__class__.__name__)
        self._image = None
        self._annotated_image = None
        self._image_labels = []
        self._boxes = None
        self._classes = None
        self._scores = None
        self.count = 0

    def setup(self):
        self._logger.info('loading cartoon dataset...')
        self._dataset.setup()
        self._logger.info('Done')
        self._sketcher = SketchGizeh()
        self._sketcher.setup()
        self._logger.info('loading tensorflow model...')
        self._image_processor.setup()
        self._logger.info('Done')
        self._path = Path(__file__).parent / '..' / '..' / 'images'
        if not self._path.exists():
            self._path.mkdir()
        self.count = len(list(self._path.glob('image*.jpg')))
        self._logger.info('setup finished.')

    def capture(self, device):
        self._image_path = Path(os.getcwd()) / "placeholder"

        self._logger.info('capturing image')
        cap = cv2.VideoCapture(device)
        ret, frame = cap.read()
        cap.release()

        return frame

    def read(self, path):
        self._image_path = Path(path)
        return self._image_processor.load_image_into_numpy_array(path)

    def process(self, img, threshold=0.3, top_x=None, debug=False):
        """processes an image. 

        :param image_path: image to process
        :param top_x: If not none, only the top X results are drawn (overrides threshold)
        :param float threshold: threshold for object detection (0.0 to 1.0)
        :param path: directory to save results to
        
        :return:
        """
        self._logger.info('processing image...')
        try:
            img_scaled = cv2.resize(img, (0,0), fx=300 / max(img.shape), fy=300 / max(img.shape))                 
            
            # detect objects
            process_time = time.time()
            self._boxes, self._scores, self._classes, num = self._image_processor.detect(img_scaled)
            process_time = time.time() - process_time

            # annotate the original image
            annotate_time = time.time()
            self._annotated_image = self._image_processor.annotate_image(img, self._boxes, self._classes, self._scores, threshold=threshold)
            annotate_time = time.time() - annotate_time

            self._sketcher = SketchGizeh()
            self._sketcher.setup(img.shape[1], img.shape[0])

            if top_x:
                sorted_scores = sorted(self._scores.flatten())
                threshold = sorted_scores[-min([top_x, self._scores.size])]

            if debug:
                scores_classes = [(self._image_processor.labels[i]["name"], score) 
                                for (i, score) in zip(np.squeeze(self._classes),np.squeeze(self._scores))]
                print("Top 10:")
                for i in range(10):
                    print(scores_classes[i]) 

            draw_time = time.time()
            self._image_labels = self._sketcher.draw_object_recognition_results(np.squeeze(self._boxes),
                                   np.squeeze(self._classes).astype(np.int32),
                                   np.squeeze(self._scores),
                                   self._image_processor.labels,
                                   self._dataset,
                                   threshold=threshold)
            draw_time = time.time() - draw_time

            if debug:
                print("Detecting %.2fs" % process_time)
                print("Annotating %.2fs" % annotate_time)
                print("Drawing %.2fs" % draw_time)
                print("Accepted", self._image_labels)

        except (ValueError, IOError) as e:
            self._logger.exception(e)

    def save_results(self, name, debug=False):
        """save result images as png and list of detected objects as txt
        if debug is true, save a list of all detected objects and their scores

        :return tuple: (path to annotated image, path to cartoon image)
        """
        self._logger.info('saving results...')            
        cartoon_path = self._image_path.with_name(name)
        self._sketcher.save_png(cartoon_path)

        if debug:
            labels_path = self._image_path.with_name('labels' + str(self.count) + '.txt')
            with open(str(labels_path), 'w') as f:
                f.writelines(self.image_labels)
            scores_path = self._image_path.with_name('scores' + str(self.count) + '.txt')
            with open(str(scores_path), 'w') as f:
                fcsv = writer(f)
                fcsv.writerow(map(str, self._scores.flatten()))
            annotated_path = self._image_path.with_name('annotated.png')
            self._save_3d_numpy_array_as_png(self._annotated_image, annotated_path)
        
        return cartoon_path

    def get_npimages(self):
        return self._sketcher.get_npimage(), self._annotated_image

    def _save_3d_numpy_array_as_png(self, img, path):
        """saves a NxNx3 8 bit numpy array as a png image

        :param img: N.N.3 numpy array
        :param path: path to save image to, e.g. './img/img.png
        :return:
        """
        if len(img.shape) != 3 or img.dtype is not np.dtype('uint8'):
            raise TypeError('image must be NxNx3 array')
        with open(str(path), 'wb') as f:
            writer = png.Writer(img.shape[1], img.shape[0], greyscale=False, bitdepth=8)
            writer.write(f, np.reshape(img, (-1, img.shape[1] * img.shape[2])))

    def close(self):
        self._image_processor.close()

    @property
    def image_labels(self):
        return self._image_labels
