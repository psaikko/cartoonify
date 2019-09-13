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
import click 

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

@click.command()
@click.option("--device", default=0, help="Device id of video camera")
@click.option("--ifile", default=None, help="Input file, uses camera if not specified")
@click.option("--ofile", default=None, help="Output file, displays window if not specified")
@click.option("--interval", default=1, help="In camera mode, time between frames")
def run(device,ifile,ofile,interval):
    app = Workflow(dataset, imageprocessor)
    app.setup()

    while True:
        if ifile != None:
            frame = app.read(ifile)
        else: # camera
            frame = app.capture(device)

        app.process(frame, threshold=0.37)
        sketch, annotated = app.get_npimages()

        if ofile != None:
            app.save_results(ofile)
            if ifile != None:
                break
            else: # camera
                time.sleep(interval)
        else: # window
            cv2.imshow('frame', sketch)
            if ifile != None:
                cv2.waitKey(0)
                break
            else: # camera
                if cv2.waitKey(interval*1000) & 0xFF == ord('q'): break
                time.sleep(interval)

    app.close()

if __name__=='__main__':
    run()
    sys.exit()
