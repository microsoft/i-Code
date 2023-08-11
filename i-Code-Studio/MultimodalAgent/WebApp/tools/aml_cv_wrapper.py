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

import datetime
import time
import requests

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

import cv2


class ComputerVisionWrapper():
    def __init__(self):
        """
        Authenticate
        Authenticates your credentials and creates a client.
        """
        # TODO: replace the correct key from Azure CV service "multimodalitycvicode" with "subscription_key_of_multimodalitycvicode"
        subscription_key = "e7631c6324034caf82f704607bb25c11"
        endpoint = "https://multimodalitycvicode.cognitiveservices.azure.com/"
        self.computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

        self.cust_endpoint = "https://multimodalitycvicode.cognitiveservices.azure.com/vision/v4.0-preview.1/operations/imageanalysis:analyze?visualFeatures=customModel&customModel-modelName=odv1"
        self.session = requests.Session()
        self.session.headers.update({'Ocp-Apim-Subscription-Key': 'e7631c6324034caf82f704607bb25c11'})
        self.session.headers.update({'Content-Type': 'application/octet-stream'})
        
        self.webcam = cv2.VideoCapture(0)

    def __del__(self):
        print('Destructor called, ComputerVisionWrapper deleted.')
        print("Turning off camera.")
        self.webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()

    def get_vision_signals(self):
        check, frame = self.webcam.read()
        ret, img_encode = cv2.imencode('.jpg', frame)
        str_encode = img_encode.tobytes()

        #image_data = open(image_path, "rb").read()
        res = self.session.post(self.cust_endpoint, data=str_encode)
        analysis = res.json()
        objects = analysis['customModelResult']['objects'][0]['classifications'][0]['label']

        img_byteio = BytesIO(str_encode)
        reader = BufferedReader(img_byteio)
        # Frame analysis
        remote_image_features = ['description']
        results_remote = self.computervision_client.analyze_image_in_stream(reader, remote_image_features)

        # Describe image
        # Get the captions (descriptions) from the response, with confidence level
        text = ""
        for caption in results_remote.description.captions:
            text += caption.text

        return {"caption": text, "objects": objects}

    def getCaption(self):
        print("===== Analyze an image - remote =====")

        # Select the visual feature(s) you want.
        remote_image_features = ['description']

        try:
            # save 1 picture per second
            check, frame = self.webcam.read()
            print(check)  # prints true as long as the webcam is running
            # print(frame)  # prints matrix values of each framecd

            if check:
                print("Get the frame.")
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
                # Return tags (need to add 'tag' in remote_image_features)
                # Print results with confidence score
                print("Tags in the remote image: ")
                if (len(results_remote.tags) == 0):
                    print("No tags detected.")
                else:
                    for tag in results_remote.tags:
                        print("'{caption.text}' with confidence {:.2f}%".format(tag.name, tag.confidence * 100))
                '''

                print(f"time elapse = {time_diff} s")

                # sleep seconds before starting next loop
                # time.sleep(time_interval)

                '''
                post the recognized result back to conversation 
                
                url = 'http://10.172.44.96:8080/api/conversation'
                myobj = {'convText': f'{caption.text}'}
                x = requests.post(url, json=myobj)
                print("######for here is the response of the " + x.text)
                '''

                return caption.text

            else:
                print("Can't get the frame, please Check your camera device")
                return ''


        except Exception as err:
            print(f"Failed to getCaption with error: {err}, ")

    def gen_frames(self):
        while True:
            success, frame = self.webcam.read()  # read the camera frame
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


if __name__ == '__main__':
    cv = ComputerVisionWrapper()
    #cv.custom_post()
    print(cv.get_vision_signals())








