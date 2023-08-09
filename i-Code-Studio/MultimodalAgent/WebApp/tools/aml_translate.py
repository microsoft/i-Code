from datetime import datetime
import json
import os
import time
import sys
import uuid
import requests
import atexit
import hashlib
import logging
from datetime import datetime

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
        format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p %Z")

class MachineTranslator():
    def __init__(self):
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')
        config = json.load(open(config_path, 'r'))['translate']

        endpoint = 'https://api.cognitive.microsofttranslator.com/'

        key, endpoint = self._setup_translation_keys(config['key'], endpoint)
        self.key = key
        self.endpoint = endpoint

    def __call__(self, text_list, src_lang=None, tgt_langs=['zh']):        
        # https://docs.microsoft.com/en-us/azure/cognitive-services/translator/language-support

        if isinstance(text_list, str):
            text_list = [text_list]

        output = [{lang: {"text":"", "alignment":{}} for lang in tgt_langs} for _ in range(len(text_list))]
        
        for idx, text in enumerate(text_list):
            if text is None or not len(text.strip()):
                continue

            result = self._translate(text, tgt_langs, self.key, self.endpoint, src_lang=src_lang, use_html=False)
            if len(result) > 0 and len(result[0]["translations"]) > 0:
                for trans in result[0]["translations"]:
                    if len(trans) > 0:
                        tgt_lang = trans["to"]
                        trans_text = trans.get("text", "")
                        trans_alignment = trans.get("alignment", {})

                        output[idx][tgt_lang]["text"] = trans_text
                        output[idx][tgt_lang]["alignment"] = trans_alignment

        if len(output) == 1:
            return output[0]

        return output

    def _setup_translation_keys(self, translator_subscription_key=None,
                            translator_endpoint=None):
        """
        Add the translation keys either from the OS environment
        or the command line args.
        Need to have a subscription key and target endpoint.
        """
        #
        key_var_name = 'TRANSLATOR_TEXT_SUBSCRIPTION_KEY'
        if key_var_name not in os.environ and translator_subscription_key is None:
            raise Exception(
                "Please set {} in the os environment or "
                "pass as an argument.".format(key_var_name))
        elif translator_subscription_key is None and key_var_name in os.environ:
            translator_subscription_key = os.environ[key_var_name]
        endpoint_var_name = 'TRANSLATOR_TEXT_ENDPOINT'
        if endpoint_var_name not in os.environ and translator_endpoint is None:
            raise Exception(
                "Please set {} in the os environment or pass as"
                "an argument.".format(
                    endpoint_var_name))
        elif translator_endpoint is None and endpoint_var_name in os.environ:
            translator_endpoint = os.environ[key_var_name]
        return translator_subscription_key, translator_endpoint

    def _translate(self, txt, targets, translation_key=None, translation_endpoint=None,
                src_lang=None, include_alignment=True, use_html=False):
        if not isinstance(targets, list):
            targets = [targets]
        assert targets, print(targets)
        if translation_key is None or translation_endpoint is None \
                or not bool(translation_key) or not bool(translation_endpoint):
            raise Exception(
                'Please include the necessary translation key and endpoint.'
                'Got: {} and {}'.format(
                    translation_key, translation_endpoint))
        subscription_key = translation_key

        endpoint = translation_endpoint
        path = '/translate?api-version=3.0'
        params = '&to={}'.format(targets[0])
        if len(targets) > 1:
            params += ''.join(['&to={}'.format(lang) for lang in targets[1:]])
        if include_alignment:
            params += "&includeAlignment=true"
        if use_html:
            params += "&textType=html"
        if src_lang is not None:
            params += '&from=%s' % src_lang
        constructed_url = endpoint + path + params
        headers = {
            'Ocp-Apim-Subscription-Key': subscription_key,
            'Ocp-Apim-Subscription-Region': 'westus',
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }
        body = [{
            'text': txt
        }]
        request = requests.post(constructed_url, headers=headers, json=body)
        if request.status_code == 200:
            return request.json()
        if request.json()['error']['code'] == 429001:
            raise ConnectionError(
                "Requests throttled. Need to wait. Body: {}".format(
                    request.json()))
        if request.json()['error']['code'] == 400077:
            num = 1000
            if len(txt) <= 1000:
                num = int(len(txt) * 0.8)
            logging.info("The maximum request size has been exceeded."
                "Size: {} Truncating to {} tokens."
                "Body: {}\nFor reference, Text={}".format(
                    len(txt), num, request.json(), txt), file=sys.stderr)
            return self.translate(txt[:num], targets, translation_key, translation_endpoint)
        raise Exception(
            "Request failed. Request response: {} Body: {}".format(
                request.json(), body))

if __name__ == '__main__':
    mt = MachineTranslator()

    tgt_langs = ["zh-Hans"]
    relations = ["Hello, my name is XXX"]
    res = mt(relations, src_lang='en', tgt_langs=tgt_langs)
    print(res)
