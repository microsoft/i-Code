# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:59:44 2022

@author: mkhademi
"""

from revChatGPT.revChatGPT import Chatbot

config = {
    "email": "mkhademi@student.ubc.ca",
    "password": "AMamamam3173"#,
    #"session_token": "<SESSION_TOKEN>", # Deprecated. Use only if you encounter captcha with email/password
    #"proxy": "<HTTP/HTTPS_PROXY>"
}

chatbot = Chatbot(config, conversation_id=None)
chatbot.refresh_session() # You need to log in on the first run
tmp = chatbot.get_chat_response(prompt="Hi! How are you?", output="text")
print(tmp)
#tmp = chatbot.get_chat_response(prompt="What is FIFA?", output="text")
#print(tmp)
response = chatbot.get_chat_response(prompt="Please answer in just one sentence what is FIFA?", output="text")
print(response)