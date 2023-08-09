import os
import openai
import json
import atexit
import hashlib
import pickle
from datetime import datetime
from tqdm import tqdm

#openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = "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"


class GPT3Generator():
    def __init__(self, engine='text-davinci-002'):
        self.engine = engine

    def __call__(self, prompt):
        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            temperature=0.9,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.6,
            stop=[" Human:", " AI:"],
            )
        return response["choices"][0]["text"].strip()

if __name__ == '__main__':
    gpt3 = GPT3Generator()

    prompt = "GPT3: Generative Pre-trained Transformer 3 (GPT-3; stylized GPT·3) is an autoregressive language model that uses deep learning to produce human-like text. \n\n iCode: iCode is a self-supervised pretraining framework where users may flexibly combine the modalities of vision, speech, and language into unified and general-purpose vector representations. \n\n Multilingual Knowledge: "

    print(gpt3(prompt))