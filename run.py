from __future__ import division
from app.workflow import Workflow
from app.drawing_dataset import DrawingDataset
from app.image_processor import ImageProcessor, tensorflow_model_name, model_path
from app.sketch import SketchGizeh
from pathlib import Path
from os.path import join
import logging
import datetime
import importlib
import sys
import time
import cv2

root = Path(__file__).parent

# init objects
dataset = DrawingDataset(str(root / 'downloads/drawing_dataset'), str(root / 'app/label_mapping.jsonl'))
imageprocessor = ImageProcessor(str(model_path),
                                str(root / 'app' / 'object_detection' / 'data' / 'mscoco_label_map.pbtxt'),
                                tensorflow_model_name)

# configure logging
logging_filename = datetime.datetime.now().strftime('%Y%m%d-%H%M.log')
logging_path = Path(__file__).parent / 'logs'
if not logging_path.exists():
    logging_path.mkdir()
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG, filename=str(Path(__file__).parent / 'logs' / logging_filename))

def run():
    app = Workflow(dataset, imageprocessor)
    app.setup()

    #path = Path(input("enter the filepath of the image to process: "))
    path = "tmp.jpg"
    app.capture(path)
    
    start = time.time()
    app.process(str(path), top_x=10)
    print("processed in %f seconds" % (time.time()-start))

    app.save_results(debug=True)
    
    app.close()

if __name__=='__main__':
    run()
    sys.exit()
