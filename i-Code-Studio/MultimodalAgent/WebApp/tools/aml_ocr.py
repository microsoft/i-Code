from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
import os
import sys
import json
import time

class AzureOCR():
    def __init__(self):
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')
        config = json.load(open(config_path, 'r'))['ocr']


        self.client = ComputerVisionClient(config["endpoint"], CognitiveServicesCredentials(config["key"]))

    def __call__(self, path, is_url=False):
        #read_image_url = "https://raw.githubusercontent.com/MicrosoftDocs/azure-docs/master/articles/cognitive-services/Computer-vision/Images/readsample.jpg"

        if is_url:
            read_response = self.client.read(read_image_url,  raw=True)
        else:
            read_image = open(path, "rb")
            read_response = self.client.read_in_stream(read_image,  raw=True)

        # Get the operation location (URL with an ID at the end) from the response
        read_operation_location = read_response.headers["Operation-Location"]
        # Grab the ID from the URL
        operation_id = read_operation_location.split("/")[-1]

        # Call the "GET" API and wait for it to retrieve the results 
        while True:
            read_result = self.client.get_read_result(operation_id)
            if read_result.status not in ['notStarted', 'running']:
                break
            time.sleep(1)

        # Print the detected text, line by line
        output = []
        if read_result.status == OperationStatusCodes.succeeded:
            for text_result in read_result.analyze_result.read_results:
                for line in text_result.lines:
                    output.append({"text": line.text, "box": line.bounding_box})

        return output
    
if __name__ == '__main__':
    ocr = AzureOCR()

    video_dir = "/data1/yuwfan/icode-studio/frames/"
    all_res = {}
    for fn in os.listdir(video_dir):
        print(fn)
        full_fn = os.path.join(video_dir, fn)
        res = ocr(full_fn)
        all_res[full_fn] = res

    json.dump(all_res, open("ocr.json", "w"))
