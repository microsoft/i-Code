#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

import logging
import sys
import os
import requests
import time
import json

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    print("""
    Importing the Speech SDK for Python failed.
    Refer to
    https://docs.microsoft.com/azure/cognitive-services/speech-service/quickstart-python for
    installation instructions.
    """)
    import sys
    sys.exit(1)

class SpeechToText():
    def __init__(self):
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')
        config = json.load(open(config_path, 'r'))['transcribe']

        # Your subscription key and region for the speech service
        self.name = "iCode Studio"
        self.description = "Transcribe service for iCode studio"


        auto_detect_source_language_config = \
            speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=["zh-CN", "en-US"])
        self.speech_config = speechsdk.SpeechConfig(subscription=config['key'], region=config['region'])

        # Sets the Priority (optional, defaults to 'Latency'). Either 'Latency' or 'Accuracy' is accepted.
        self.speech_config.set_property(
            property_id=speechsdk.PropertyId.SpeechServiceConnection_SingleLanguageIdPriority, value='Latency')

        self.source_language_recognizer = speechsdk.SourceLanguageRecognizer(
                                        speech_config=self.speech_config,
                                        auto_detect_source_language_config=auto_detect_source_language_config)

        self.speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config)

        logging.info("Starting transcription client...")
        
    def __call__(self):
        """performs one-shot speech recognition from the default microphone"""
        # Creates a speech recognizer using microphone as audio input.
        # The default language is "en-us".
        print("START ASR")
        #future = self.source_language_recognizer.recognize_once_async()
        result = self.speech_recognizer.recognize_once()
        #result = future.get()
        # # Check the result
        # if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        #     print("RECOGNIZED: {}".format(result))
        #     detected_src_lang = result.properties[
        #         speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult]
        #     print("Detected Language: {}".format(detected_src_lang))
        # elif result.reason == speechsdk.ResultReason.NoMatch:
        #     print("No speech could be recognized: {}".format(result.no_match_details))
        # elif result.reason == speechsdk.ResultReason.Canceled:
        #     cancellation_details = result.cancellation_details
        #     print("Speech Language Detection canceled: {}".format(cancellation_details.reason))
        #     if cancellation_details.reason == speechsdk.CancellationReason.Error:
        #         print("Error details: {}".format(cancellation_details.error_details))

        # Check the result
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print("Recognized: {}".format(result.text))
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("Speech Recognition canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))

        return result.text, result.reason

if __name__ == '__main__':
    asr = SpeechToText()
    print(asr())

