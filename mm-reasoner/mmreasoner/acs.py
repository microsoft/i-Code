import json
import os
import time

import azure.ai.vision as sdk
import numpy as np


def process_dataset(dataset, split, service_options, analysis_options):
    if dataset == 'OK-VQA':
        with open(f"mscoco_{split}_annotations.json", 'r') as f:
            anns = json.load(f)
        image_names = set()
        for i in range(len(anns['annotations'])):
            image_id = anns['annotations'][i]['image_id']
            image_names.add('COCO_' + split + '_' + str(image_id).zfill(12) + '.jpg')
        image_names = sorted(list(image_names))

    if dataset == 'A-OKVQA':
        image_names = sorted(os.listdir("A-OKVQA-images"))  

    if dataset == 'KVQA':
        image_names = sorted(os.listdir("KVQAimgs-resized"))  
        image_dir = "KVQAimgs-resized/"

    if dataset == 'GQA':
        image_names = sorted(os.listdir("images"))  
        image_dir = "images/"

    if dataset == 'FVQA':
        image_names = sorted(os.listdir("images"))  
        image_dir = "images/"

    if dataset == 'VQAv2':
        image_names = sorted(os.listdir(f"Images/{split}/"))  
        image_dir = f"Images/{split}/"

    if dataset == 'Abstract_Scenes':
        image_names = sorted(os.listdir(f"images/"))  
        image_dir = f"images/"

    i = 0
    while i < len(image_names):
        vision_source = sdk.VisionSource(filename=image_dir + image_names[i])
        image_analyzer = sdk.ImageAnalyzer(service_options, vision_source, analysis_options)

        result_dict = dict()
        result = image_analyzer.analyze()
        
        if result.reason == sdk.ImageAnalysisResultReason.ANALYZED:
            if result.caption is not None:
                cap = dict()
                cap['caption'] = result.caption.content
                cap['confidence'] = np.round(result.caption.confidence, 3)
                result_dict['caption'] = cap

            if result.text is not None:
                text_list = []
                for line in result.text.lines:
                    line_dict = dict()
                    points_string = "{" + ", ".join([str(int(point)) for point in line.bounding_polygon]) + "}"
                    line_dict["bounding_polygon"] = line.bounding_polygon
                    line_dict["content"] = line.content
                    words_list = []
                    for word in line.words:
                        word_dict = dict()
                        word_dict["content"] = word.content
                        word_dict["confidence"] = np.round(word.confidence, 3)
                        word_dict["bounding_polygon"] = word.bounding_polygon
                        words_list.append(word_dict.copy())
                        points_string = "{" + ", ".join([str(int(point)) for point in word.bounding_polygon]) + "}"
                    line_dict["words"] = words_list.copy()
                    text_list.append(line_dict.copy())
                result_dict['text'] = text_list.copy()
            if result.objects is not None:
                objects = []
                for j in range(len(result.objects)):
                    obj = dict()
                    obj['name'] = result.objects[j].name
                    obj['confidence'] = np.round(result.objects[j].confidence, 3)
                    x = result.objects[j].bounding_box.x
                    y = result.objects[j].bounding_box.y
                    w = result.objects[j].bounding_box.w
                    h = result.objects[j].bounding_box.h
                    obj['bounding_box'] = [x, y, w, h]
                    objects.append(obj.copy())
                result_dict['objects'] = objects.copy()
            if result.tags is not None:
                tags = []
                if (len(result.tags) == 0):
                    print("No tags detected.")
                else:
                    for tag in result.tags:
                        tag_dict = dict()
                        tag_dict['tag_name'] = tag.name
                        tag_dict['confidence'] = np.round(tag.confidence, 3)
                        tags.append(tag_dict.copy())
                result_dict['tags'] = tags.copy()
            if result.dense_captions is not None:
                dense_captions = []
                for j in range(len(result.dense_captions)):
                    dense_cap = dict()
                    dense_cap['content'] = result.dense_captions[j].content
                    dense_cap['confidence'] = np.round(result.dense_captions[j].confidence, 3)
                    x = result.dense_captions[j].bounding_box.x
                    y = result.dense_captions[j].bounding_box.y
                    w = result.dense_captions[j].bounding_box.w
                    h = result.dense_captions[j].bounding_box.h
                    dense_cap['bounding_box'] = [x, y, w, h]
                    dense_captions.append(dense_cap.copy())
                result_dict['dense_captions'] = dense_captions.copy()
            if result.people is not None:
                    people = []
                    for j in range(len(result.people)):
                        person = dict()
                        person['confidence'] = np.round(result.people[j].confidence, 3)
                        x = result.people[j].bounding_box.x
                        y = result.people[j].bounding_box.y
                        w = result.people[j].bounding_box.w
                        h = result.people[j].bounding_box.h
                        person['bounding_box'] = [x, y, w, h]
                        people.append(person.copy())
                    result_dict['people'] = people.copy()
            if dataset == 'OK-VQA':
                with open("acs-ok-vqa/" + image_names[i][:-4] + ".json", "w") as f:
                    json.dump(result_dict.copy(), f)
                    
            if dataset == 'A-OKVQA':
                with open("acs-aok-vqa/" + image_names[i][:-4] + ".json", "w") as f:
                    json.dump(result_dict.copy(), f)
                    
            if dataset == 'KVQA':
                with open("acs-kvqa/" + image_names[i][:-4] + ".json", "w") as f:
                    json.dump(result_dict.copy(), f)
            
            if dataset == 'GQA':
                with open("acs-gqa/" + image_names[i][:-4] + ".json", "w") as f:
                    json.dump(result_dict.copy(), f)
                    
            if dataset == 'FVQA':
                with open("acs-fvqa/" + image_names[i][:-4] + ".json", "w") as f:
                    json.dump(result_dict.copy(), f)
                    
            if dataset == 'VQAv2':
                with open(f"acs-{split}/" + image_names[i][:-4] + ".json", "w") as f:
                    json.dump(result_dict.copy(), f)
                    
            if dataset == 'Abstract_Scenes':
                with open(f"acs-abstract-scenes/" + image_names[i][:-4] + ".json", "w") as f:
                    json.dump(result_dict.copy(), f)
            print(i)
            i += 1      
        elif result.reason == sdk.ImageAnalysisResultReason.ERROR:
        
            error_details = sdk.ImageAnalysisErrorDetails.from_result(result)
            print(" Analysis failed.")
            print("   Error reason: {}".format(error_details.reason))
            print("   Error code: {}".format(error_details.error_code))
            print("   Error message: {}".format(error_details.message))
            time.sleep(2)
            continue

def main():
    endpoint = "https://<YOUR_ENDPOINT>.cognitiveservices.azure.com/"
    subscription_key = ""
    service_options = sdk.VisionServiceOptions(endpoint, subscription_key)

    analysis_options = sdk.ImageAnalysisOptions()

    analysis_options.features = (
        sdk.ImageAnalysisFeature.CAPTION |
        sdk.ImageAnalysisFeature.TEXT |
        sdk.ImageAnalysisFeature.OBJECTS |
        sdk.ImageAnalysisFeature.PEOPLE |
        sdk.ImageAnalysisFeature.TAGS |
        sdk.ImageAnalysisFeature.DENSE_CAPTIONS
    )

    analysis_options.language = "en"
    analysis_options.gender_neutral_caption = True

    datasets = [
        'Abstract_Scenes',
        'VQAv2',
        'FVQA',
        'GQA',
        'KVQA',
        'A-OKVQA',
        'OK-VQA'
    ]

    split = "val2014"
    # split = "val2014" # 5033 OK-VQA images, 5046 questions
    # split = "train2014" # 8998 OK-VQA images, 9009 questions

    for dataset in datasets:
        process_dataset(dataset, split, service_options, analysis_options)


if __name__ == '__main__':
    main()
