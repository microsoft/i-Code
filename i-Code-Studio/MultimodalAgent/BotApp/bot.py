# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import ChannelAccount

from tools.openai_gpt3 import GPT3Generator

class MyBot(ActivityHandler):
    # See https://aka.ms/about-bot-activity-message to learn more about the message and other activity types.
    def __init__(self):
        self.context = json.loads('{}')
        self.gpt3 = GPT3Generator()

    async def on_message_activity(self, turn_context: TurnContext):
        start = "How can I help you today?"

        def format_gpt3_prompt(context, query):
            prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n"

            if not context:
                return prompt + f"\nAI:{start}\nHuman: {query}"
            else:
                return prompt + f"\n{context['history']}\nHuman: {query}"

        query = turn_context.activity.text

        print(f"query: {query}")
        if not query:
            response = start
            history = f"AI: {start}"
        else:
            prompt = format_gpt3_prompt(self.context, query)
            response = self.gpt3(prompt)
            if response.startswith("AI: "):
                response = response[4:]

            history = f"\nHuman: {query}\nAI: {response}"

        if not self.context:
            self.context['history'] = history
        else:
            self.context['history'] += history

        # await turn_context.send_activity(f"You said '{ turn_context.activity.text }'")
        await turn_context.send_activity(f"{ response}")

    async def on_members_added_activity(
        self,
        members_added: ChannelAccount,
        turn_context: TurnContext
    ):
        for member_added in members_added:
            if member_added.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("Hello, I am Cortana V3 created by Azure Cognitive Services. How can I help you today?")
