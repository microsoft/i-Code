'''
Computer Vision Quickstart for Microsoft Azure Cognitive Services. 
Uses local and remote images in each example.
Prerequisites:
    - Install the Computer Vision SDK:
      pip install --upgrade azure-cognitiveservices-vision-computervision
    - Install PIL:
      pip install --upgrade pillow
    - Create folder and collect images: 
      Create a folder called "images" in the same folder as this script.
      Go to this website to download images:
        https://github.com/Azure-Samples/cognitive-services-sample-data-files/tree/master/ComputerVision/Images
      Add the following 7 images (or use your own) to your "images" folder: 
        faces.jpg, gray-shirt-logo.jpg, handwritten_text.jpg, landmark.jpg, 
        objects.jpg, printed_text.jpg and type-image.jpg
Run the entire file to demonstrate the following examples:
    - Describe Image
    - Categorize Image
    - Tag Image
References:
    - SDK: https://docs.microsoft.com/en-us/python/api/azure-cognitiveservices-vision-computervision/azure.cognitiveservices.vision.computervision?view=azure-python
    - Documentaion: https://docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/index
    - API: https://westus.dev.cognitive.microsoft.com/docs/services/computer-vision-v3-2/operations/5d986960601faab4bf452005
'''

from array import array
from io import BytesIO, BufferedReader

import datetime;
import time

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

import cv2

class ComputerVisionWrapper:
    def __init__(self):
        self.engine = engine
        '''
        Authenticate
        Authenticates your credentials and creates a client.
        '''
        subscription_key = "subscription_key_of_multimodalitycvicode"
        endpoint = "https://multimodalitycvicode.cognitiveservices.azure.com/"
        self.computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
        '''
        END - Authenticate
        '''

print("===== Analyze an image - remote =====")
# Select the visual feature(s) you want.
remote_image_features = ['description','tags']

webcam = cv2.VideoCapture(0)
while True:
    try:
        # save 1 picture per second
        check, frame = webcam.read()
        print(check)  # prints true as long as the webcam is running
        # print(frame)  # prints matrix values of each framecd 

        if check: 
            # cv2.imshow("Capturing", frame)
            # timestamp = datetime.datetime.now().timestamp()
            # cv2.imwrite(filename=f'saved_img_{timestamp}.jpg', img=frame)
            # print(f"Image: saved_img_{timestamp}.jpg saved!")

            # Encode a frame to jpeg stream
            ret, img_encode = cv2.imencode('.jpg', frame)
            str_encode = img_encode.tobytes()
            img_byteio = BytesIO(str_encode)
            img_byteio.name = 'img.jpg'
            reader = BufferedReader(img_byteio)

            # Frame analysis
            time_start = time.time()
            results_remote = self.computervision_client.analyze_image_in_stream(reader, remote_image_features)
            time_diff = time.time() - time_start

            # Describe image
            # Get the captions (descriptions) from the response, with confidence level
            print("Description of remote image: ")
            if ( not results_remote.description):
                print("No description detected.")
            else:
                for caption in results_remote.description.captions:
                    print("'{}' with confidence {:.2f}%".format(caption.text, caption.confidence * 100))
            print()
            
            '''
            # Return tags
            # Print results with confidence score
            print("Tags in the remote image: ")
            if (len(results_remote.tags) == 0):
                print("No tags detected.")
            else:
                for tag in results_remote.tags:
                    print("'{}' with confidence {:.2f}%".format(tag.name, tag.confidence * 100))
            '''

            print(f"time elapse = {time_diff} s")
 
            # sleep seconds before starting next loop
            time.sleep(5)
        
    # "ctrl + c" to exit the program                                                                
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break









