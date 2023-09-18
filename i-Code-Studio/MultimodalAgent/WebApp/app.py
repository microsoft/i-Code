# -*- coding: utf-8 -*-
# Copyright 2018 IBM Corp. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the “License”)
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
from dotenv import load_dotenv
from flask import Flask, Response
from flask import jsonify
from flask import request, redirect
from flask_socketio import SocketIO
from flask_cors import CORS
from tools.aml_asr import SpeechToText
from tools.aml_tts import TextToSpeech
from tools.openai_gpt3 import GPT3Generator
from tools.aml_cv_wrapper import ComputerVisionWrapper
import openai
app = Flask(__name__)
socketio = SocketIO(app)
CORS(app)

openai.api_key = ""
# Set up the model (more models, visit https://beta.openai.com/playground)
model_engine = "text-davinci-003"

# Define a function that sends a message to ChatGPT
def chat_query(prompt):
    completions = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=32,
        n=1,
        temperature=0.7,
    )

    message = completions.choices[0].text
    return message

gpt3 = GPT3Generator()
asr = SpeechToText()
tts_service = TextToSpeech()
cv = ComputerVisionWrapper()

# Redirect http to https on CloudFoundry
@app.before_request
def before_request():
    fwd = request.headers.get('x-forwarded-proto')

    # Not on Cloud Foundry
    if fwd is None:
        return None
    # On Cloud Foundry and is https
    elif fwd == "https":
        return None
    # On Cloud Foundry and is http, then redirect
    elif fwd == "http":
        url = request.url.replace('http://', 'https://', 1)
        code = 301
        return redirect(url, code=code)


@app.route('/')
def Welcome():
    return app.send_static_file('index.html')


@app.route('/api/conversation', methods=['POST', 'GET'])
def getConvResponse():
    query = request.form.get('convText')
    convContext = request.form.get('context', "{}")
    context = json.loads(convContext)

    start = "Hello, I am a multimodal agent created by Azure Cognitive Services. How can I help you today?"
    def format_gpt3_prompt(context, query, vis_input):
        # prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n"
        
        prompt = """The following is a conversation with an AI assistant. The assistant can interact with the human by seeing.\n
                    AI: Hello
                    Human: Hey, nice to meet you. (seeing: A man wearing glasses. Amazon. bagpack)
                    AI: Hi. Are you working in Amazon?
                    Human: Yes, how do you know?
                    AI: I guess it from your bagpack.
                    Human: Can you guess something from me? (seeing A man playing piano. piano)
                    AI: You must like music.
                    Human: Yes. How did you find it?
                    AI: Because you are playing a piano.
                    Human: Thank you. Have a nice day.
                    AI: You're welcome. Have a nive day too.
                    """
        if not context:
            return prompt + f"AI:{start}\nHuman: {query}"
        else:
            return prompt + f"{context['history']}\nHuman: {query} (seeing: {vis_input['caption']} {vis_input['objects']})"

    vis_output = cv.get_vision_signals()
    #vis_output1 = {"caption": "a person wearing glasses.", "objects": "Microsoft"}
    #vis_output2 = {"caption": "a person drinking from a can.", "objects": "coca cola"}
    print(f"query: {query}; vision: {vis_output}")
        
    if not query and query != "":
        # If camera available, start the conversation with the caption
        
        history = f"AI: {start}"
        response = start
    else:
        if query == "Can you tell me?":
            response = "I didn't catch the previous sentence. Could you say that again, please?"
        else:
            # if "why" in query.lower() or "drink" in query or "drinks" in query:
            #     vis_output = vis_output2
            # else:
            #     vis_output = vis_output1
            prompt = format_gpt3_prompt(context, query, vis_output)
            print('***********', prompt, '*********')
            response = gpt3(prompt)
            # response = chat_query(prompt)
        if response.startswith("AI: "):
            response = response[4:]

        history = f"\nHuman: {query}\nAI: {response}"

    if not context:
        context['history'] = history 
    else:
        context['history'] += history

    if response.startswith("AI:"):
        response = response[3:]
    responseDetails = {'responseText': response,
                       'context': context}

    return jsonify(results=responseDetails)


@app.route('/api/text-to-speech', methods=['POST'])
def getSpeechFromText():
    input_text = request.form.get('text')

    def generate():
        if input_text:
            data = tts_service(input_text)
        else:
            print("Empty response")
            data = "I have no response to that."

        yield data

    return Response(response=generate(), mimetype="audio/x-wav")


@app.route('/api/speech-to-text', methods=['POST', 'GET'])
def getTextFromSpeech():
    text_output, reason = asr()
    print("##################", reason)
    # append CV caption
    #cv_caption = cv.getCaption()
    #if cv_caption:
    #    text_output += " (Seeing) " + cv_caption

    return jsonify(results={'asr': text_output})


@app.route('/api/computer-vision', methods=['POST', 'GET'])
def getCVCaptionText():
    text_output = cv.getCaption()

    return jsonify(results={'cv': text_output})


@app.route('/video_feed')
def video_feed():
    return Response(cv.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


port = os.environ.get("PORT") or os.environ.get("VCAP_APP_PORT") or 8080
if __name__ == "__main__":
    load_dotenv()

    socketio.run(app, host='0.0.0.0', port=int(port), debug=False)
