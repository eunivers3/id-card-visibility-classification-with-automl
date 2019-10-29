"""
Operationalization
Make a predict.py module and write a function that accepts the original 
RGB (3-channel) images and goes through the Feature Engineering and 
Inference pipelines to yield the predicted result.
"""

from google.cloud import automl_v1beta1 as automl
from google.cloud.automl_v1beta1.proto import service_pb2
import json, os
from cv2 import cv2

# load configurations
with open('config.json') as f:
    config = f.read()
config = json.loads(config)

project_id = config.get("PROJECT_ID")
model_id = config.get("MODEL_ID")
score_threshold = config.get("MODEL_SCORE_THRESHOLD")
# os.environ['GOOGLE_APPLICATION_CREDENTIALS']="[PATH]"

def process_image(infile_path,outfile_path):
    """convert RGB image to  1x channel, add gaussian filter and write to file"""
    from scipy.ndimage import gaussian_filter
    #convert RGB img to numpy.array
    if os.path.exists(infile_path):
        img = cv2.imread(infile_path)
        # as cv2 is BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #convert to 1x channel image (grayscale)
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # add gaussian blur to remove noise
        new_img = gaussian_filter(new_img, sigma=2)
        cv2.imwrite(outfile_path, new_img)
    else:
        print(infile_path, 'does not exist') # log err
        raise FileNotFoundError

def get_prediction(content):
    # 'content' is base-64-encoded image data.
    prediction_client = automl.PredictionServiceClient()
    name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
    payload = {'image': {'image_bytes': content }}
    params = {}
    request = prediction_client.predict(name, payload, params)
    return request 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Make a prediction on a local image')
    parser.add_argument('image_file', help='The image you\'d like to crop.')
    args = parser.parse_args()

    import tempfile
    # create temp image to predict on
    with tempfile.NamedTemporaryFile(suffix='.png') as temp:
        process_image(args.image_file, temp.name)
        with open(temp.name, 'rb') as png:
            png.seek(0)
            content = png.read()
    
    # print prediction results
    response = get_prediction(content)
    if response.payload:
        prediction = response.payload[0].display_name
        score = response.payload[0].classification.score
        results = {
            "filepath" : args.image_file,
            "visibility": prediction,
            "score" :score
        }
    else:
        results = {
            "filepath" : args.image_file,
            "visibility": None,
            "score" : None
        }
    print('<<< Prediction >>> \n',json.dumps(results, indent=4))
