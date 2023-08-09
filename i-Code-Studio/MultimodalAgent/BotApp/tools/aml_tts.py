#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
"""
Speech synthesis samples for the Microsoft Cognitive Services Speech SDK
"""

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    print("""
    Importing the Speech SDK for Python failed.
    Refer to
    https://docs.microsoft.com/azure/cognitive-services/speech-service/quickstart-text-to-speech-python for
    installation instructions.
    """)
    import sys
    sys.exit(1)

import logging
import os
import sys
import json

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
        format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p %Z")

"""performs speech synthesis and push audio output to a stream"""
class PushAudioOutputStreamSampleCallback(speechsdk.audio.PushAudioOutputStreamCallback):
    """
    Example class that implements the PushAudioOutputStreamCallback, which is used to show
    how to push output audio to a stream
    """
    def __init__(self) -> None:
        super().__init__()
        self._audio_data = bytes(0)
        self._closed = False

    def write(self, audio_buffer: memoryview) -> int:
        """
        The callback function which is invoked when the synthesizer has an output audio chunk
        to write out
        """
        self._audio_data += audio_buffer
        print("{} bytes received.".format(audio_buffer.nbytes))
        return audio_buffer.nbytes

    def close(self) -> None:
        """
        The callback function which is invoked when the synthesizer is about to close the
        stream.
        """
        self._closed = True
        print("Push audio output stream closed.")

    def get_audio_data(self) -> bytes:
        return self._audio_data

    def get_audio_size(self) -> int:
        return len(self._audio_data)

class TextToSpeech():
    def __init__(self):
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')
        config = json.load(open(config_path, 'r'))['synthesis']

        self.speech_config = speechsdk.SpeechConfig(subscription=config['key'], region=config['region'])
        self.stream_callback = PushAudioOutputStreamSampleCallback()

        push_stream = speechsdk.audio.PushAudioOutputStream(self.stream_callback)
        self.stream_config = speechsdk.audio.AudioOutputConfig(stream=push_stream)

    def __call__(self, text):
        
        # Creates audio output stream from the callback
        # Creates a speech synthesizer using push stream as audio output.
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config)
        result = speech_synthesizer.speak_text_async(text).get()

        return b'1'

        #speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=self.stream_config)
        # print("Totally {} bytes received.".format(self.stream_callback.get_audio_size()))
        # assert(result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted)
        #return self.stream_callback.get_audio_data()


if __name__ == '__main__':
    tts = TextToSpeech()
    tts(text="Hello, world")