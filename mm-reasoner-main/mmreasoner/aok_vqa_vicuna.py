import json
import nltk
import openai
import time
import os
import pickle
from PIL import Image
import numpy as np

q_ids = []
with open("aokvqa_v1p0_train.json", 'r') as f:
    aok_train = json.load(f)
q_ids += [aok_train[i]['question_id'] for i in range(len(aok_train))]
with open("aokvqa_v1p0_val.json", 'r') as f:
    aok_val = json.load(f)
q_ids += [aok_val[i]['question_id'] for i in range(len(aok_val))]
with open("aokvqa_v1p0_test.json", 'r') as f:
    aok_test = json.load(f)
q_ids += [aok_test[i]['question_id'] for i in range(len(aok_test))]

q_ids = sorted(q_ids)
aok_q_ids = dict()
for i, q_id in enumerate(q_ids):
    aok_q_ids[q_id] = i
with open('aok_q_ids.pkl', 'wb') as f:
    pickle.dump(aok_q_ids, f)

openai.api_key = "EMPTY" # Not support yet
openai.api_base = "http://localhost:8000/v1"
model = "vicuna-13b-v1.1"

def chat_query(prompt):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                'role': 'system',
                'content': 'The following is a conversation with an AI assistant. Given an image description and a question about the image, the assistant provides rationales that are required or helpful to answer the question correctly. These rationales can be common sense knowledge, external information, basic factual knowledge, supporting facts, or any information from the image that is required or helpful to answer the question. Each image is described with a set of captions, a set of objects present in the image, a set of tags associated with the image, and a set of texts written in the image. In each turn of the conversation, the human user describes an image and asks a question about the image using a dictionary in the Python language. Each question has a unique id. The AI responds with a dictionary in the Python language. The dictionary has 3 keys: "question id", "rationales", and "potential answers". If an image description lacks sufficient information, the AI must still predict a list of potential answers to the question and provide likely rationales.'
            },
            {
                'role': 'user',
                'content': 'The description of the images and the given question are: {"captions": [{"caption": "a person standing with luggage on the street"}, {"caption": "a person holding a bag"}, {"caption": "a red suv on the road"}, {"caption": "a black suitcase with brown trim"}, {"caption": "a close up of a toolbox"}, {"caption": "a person holding a red bag"}], "objects": [{"object": "Land vehicle"}, {"object": "car"}, {"object": "Luggage and bags"}, {"object": "Van"}, {"object": "suitcase"}, {"object": "person"}], "tags": [{"tag": "outdoor"}, {"tag": "person"}, {"tag": "clothing"}, {"tag": "road"}, {"tag": "wheel"}, {"tag": "luggage and bags"}, {"tag": "land vehicle"}, {"tag": "tree"}, {"tag": "car"}, {"tag": "luggage"}, {"tag": "footwear"}, {"tag": "jeans"}, {"tag": "vehicle"}, {"tag": "man"}, {"tag": "suitcase"}, {"tag": "handbag"}, {"tag": "trousers"}, {"tag": "ground"}, {"tag": "street"}, {"tag": "standing"}], "texts": [], "question": "What is the man by the bags awaiting?", "question id": "22MexNkBPpdZGX6sxbxVBH"}. Please provide the rationales and a list of potential answers for the question.'
            },
            {
                'role': 'assistant',
                'content': '{"question id": "22MexNkBPpdZGX6sxbxVBH", "potential answers": ["ride", "his ride", "traffic", "travelling", "bus", "taxi", "cab"], "rationales": "A train would not be on the street, he would not have luggage waiting for a delivery, and the skateboarder is there and not paying attention to him so a cab is the only possible answer. He has bags as if he is going someone, and he is on a road waiting for vehicle that can only be moved on the road and is big enough to hold the bags. He looks to be waiting for a paid ride to pick him up."}'
            },
            {
                'role': 'user',
                'content': 'The description of the images and the given question are: {"captions": [{"caption": "a person pointing at a plate of pizza"}, {"caption": "a person pointing at a plate of pizza"}, {"caption": "a close up of a keyboard"}, {"caption": "a close up of a mouse"}, {"caption": "a computer mouse and keyboard on a desk"}], "objects": [{"object": "Wall clock"}, {"object": "computer mouse"}, {"object": "person"}], "tags": [{"tag": "person"}, {"tag": "indoor"}, {"tag": "computer mouse"}, {"tag": "wall"}, {"tag": "clothing"}, {"tag": "man"}, {"tag": "computer"}, {"tag": "input device"}, {"tag": "computer keyboard"}, {"tag": "food"}, {"tag": "pizza"}], "texts": [{"text": "TION"}, {"text": "CAUT"}, {"text": "WARNING"}, {"text": "ZOMBIES"}, {"text": "AHEAD"}, {"text": "BS"}, {"text": "DO NOT FEE"}, {"text": "THE ZOMERE"}, {"text": "TTAC"}, {"text": "APAT"}], "question": "Where does this man eat pizza?", "question id": "22ZAvqke8EhGDj8e4eyios"}. Please provide the rationales and a list of potential answers for the question.'
            },
            {
                'role': 'assistant',
                'content': '{"question id": "22ZAvqke8EhGDj8e4eyios", "potential answers": ["work desk", "office", "desk", "at work", "at desk", "work"], "rationales": "The man is eating pizza at a work desk in an office setting. The man is near a computer keyboard. The man is sitting at an office desk at his job and eating his pizza."}'
            },
            {
                'role': 'user',
                'content': prompt
            }
        ],
        temperature=0.98,
        max_tokens=256,
        top_p=0.98,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    return completion.choices[0].message.content

batch_size = 1

i = 0
split = 'train'
if split == 'train':
    anns = aok_train
if split == 'val':
    anns = aok_val
if split == 'test':
    anns = aok_test

n = len(anns)
retries = 0
image_names = os.listdir('A-OKVQA-images')
while i < n:
    prompt_ = "The description of the images and the given question are: "
    q_ids = []
    image_ids = []
    qs = []
    item_list = []
    reply_list = []
    for j in range(i, min(i+batch_size, n)):
        q = anns[j]['question']
        q_id = anns[j]['question_id']
        answers = anns[j]['direct_answers']
        image_id = anns[j]['image_id']
        q_ids.append(q_id)
        image_ids.append(image_id)
        qs.append(q)
        if split == 'test':
            fn = f"COCO_test2015_" + str(image_id).zfill(12) + ".jpg"
        else:
            fn = f"COCO_train2014_" + str(image_id).zfill(12) + ".jpg"
            if fn not in image_names:
                fn = f"COCO_val2014_" + str(image_id).zfill(12) + ".jpg"

        image = Image.open("A-OKVQA-images/" + fn)
        width, height = image.size
        with open("acs-aok-vqa/" + fn[:-4] + ".json", 'r') as f:
            acs = json.load(f)

        dense_captions = []
        for d_cap in acs["dense_captions"]:
            dense_captions.append({"caption": d_cap["content"]})

        objects = []
        for ob in acs["objects"]:
            objects.append({"object": ob["name"]})

        tags = []
        for tg in acs["tags"]:
            tags.append({"tag": tg["tag_name"]})

        texts = []
        for tx in acs["text"]:
            texts.append({"text": tx["content"]})

        people = []
        for pe in acs["people"]:
            if pe["confidence"] > 0.5:
                x, y, w, h = pe["bounding_box"]
                bb = list(np.round([x/width, y/height, w/width, h/height], 1))
                people.append({"confidence": np.round(pe["confidence"], 1), "bounding box": bb})

        info = dict()
        info["captions"] = dense_captions
        info["objects"] = objects
        info["tags"] = tags
        info["texts"] = texts
        info["question"] = q
        info["question id"] = q_id
        info["named_entities"] = anns["named_entities"]
        item_list.append(info.copy())

    prompt_ = prompt_ + json.dumps(item_list[0])
    prompt_ += ". Please provide the rationales and a list of potential answers for the question."

    try:
        response = chat_query(prompt_)   
        res = dict()
        res['response'] = json.loads(response)
        print(res['response'])
        print(i)
        res['question'] = q
        res['image_id'] = image_id
        res['question id'] = q_id
    except:
        retries += 1
        if retries >= 10:
            i += batch_size
            retries = 0
        continue
    retries = 0
    with open("aok-vqa-gpt4-train/" + str(i) + ".json", "w") as f:
        json.dump(res, f)
    i += batch_size
