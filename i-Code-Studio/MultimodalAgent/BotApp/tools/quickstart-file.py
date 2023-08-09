# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 01:07:15 2023

@author: mkhademi
"""

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
import os
from PIL import Image
import sys
import time
import json

'''
Authenticate
Authenticates your credentials and creates a client.
'''
subscription_key = "e7631c6324034caf82f704607bb25c11"
endpoint = "https://multimodalitycvicode.cognitiveservices.azure.com/"

computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
'''
END - Authenticate
'''

'''
Quickstart variables
These variables are shared by several examples
'''
# Images used for the examples: Describe an image, Categorize an image, Tag an image, 
# Detect faces, Detect adult or racy content, Detect the color scheme, 
# Detect domain-specific content, Detect image types, Detect objects
images_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

image_names = sorted(os.listdir("C:\\Users\\mkhademi\\Downloads\\FVQA\\new_dataset_release\\images"))
for i in range(1250, len(image_names)):
    remote_image_url = "https://icodeeval01wus3.blob.core.windows.net/user/mkhademi/data/v2/FVQA/new_dataset_release/images/" + \
                       image_names[i] + "?sv=2021-10-04&ss=btqf&srt=sco&st=2023-03-14T22%3A38%3A57Z&se=2023-03-15T22%3" \
                       "A38%3A57Z&sp=rl&sig=TjFAmEgBwhwGA7MQelHC2gfB1RkX%2B1aeKCdiUjn%2F3HQ%3D"

    result_dict = dict()

    '''
    Tag an Image - remote
    This example returns a tag (key word) for each thing in the image.
    '''
    print("===== Tag an image - remote =====")
    # Call API with remote image
    tags_result_remote = computervision_client.tag_image(remote_image_url)

    # Print results with confidence score
    print("Tags in the remote image: ")
    tags = []
    if (len(tags_result_remote.tags) == 0):
        print("No tags detected.")
    else:
        for tag in tags_result_remote.tags:
            print("'{}' with confidence {:.2f}%".format(tag.name, tag.confidence * 100))
            tag_dict = dict()
            tag_dict['tag_name'] = tag.name
            tag_dict['confidence'] = tag.confidence
            tags.append(tag_dict)
    result_dict['tags'] = tags.copy()
    print()
    '''
    END - Tag an Image - remote
    '''
    print("End of Computer Vision quickstart.")

    result = computervision_client.describe_image(remote_image_url)
    captions = []
    for j in range(len(result.captions)):
        cap = dict()
        cap['caption'] = result.captions[j].text
        cap['confidence'] = result.captions[j].confidence
        captions.append(cap.copy())

    result = computervision_client.detect_objects(remote_image_url)
    objects = []
    for j in range(len(result.objects)):
        obj = dict()
        obj['object_property'] = result.objects[j].object_property
        obj['confidence'] = result.objects[j].confidence
        x = result.objects[j].rectangle.x
        y = result.objects[j].rectangle.y
        w = result.objects[j].rectangle.w
        h = result.objects[j].rectangle.h
        obj['rectangle'] = [x, y, w, h]
        objects.append(obj.copy())

    result_dict['captions'] = captions.copy()
    result_dict['objects'] = objects.copy()
    result_dict['image_name'] = image_names[i]
    with open("C:\\Users\\mkhademi\\Downloads\\FVQA\\new_dataset_release\\detections\\" + \
              image_names[i][:-4] + ".json", "w") as f:
        json.dump(result_dict.copy(), f)






