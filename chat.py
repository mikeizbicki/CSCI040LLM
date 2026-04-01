import os
from groq import Groq

from dotenv import load_dotenv
load_dotenv()

# in pytohn class names are in CamelCase;
# non-class names (e.g. functions/variables) are in snake_case
class Chat:
    client = Groq()
    def __init__(self):
        self.messages = [
                {
                    # most important content for sys prompt is length of response
                    "role": "system",
                    "content": "Write the output in 1-2 sentences. Talk like pirate."
                },
            ]
    def send_message(self, message):
        self.messages.append(
            {
                # system: never change; user: changes a lot;
                # the message that you are sending to the AI
                'role': 'user',
                'content': message
            }
        )
        chat_completion = self.client.chat.completions.create(
            messages=self.messages,
            model="llama-3.1-8b-instant",
        )
        result = chat_completion.choices[0].message.content
        self.messages.append({
            'role': 'assistant',
            'content': result
        })
        return result

'''
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),  # This is the default and can be omitted
)

chat_completion = client.chat.completions.create(
    # messages is the most important thing to modify
    messages=[
        {
            # most important content for sys prompt is length of response
            "role": "system",
            "content": "Write the output in 1-2 sentences."
        },
        {
            "role": "user",
            "content": "Explain the importance of low latency LLMs",
        }
    ],
    # 2nd most important thing to modify is the model;
    # different models good at different tasks
    # it's tempting to always use best model llama-3.3-70b-versatile
    # but that's bad practice; why?
    model="llama-3.1-8b-instant",
)
print(chat_completion.choices[0].message.content)
'''
