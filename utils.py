from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI()

# get chat completion from standard chat LLMs
def get_chat_completion(messages, model="gpt-4o", temperature=0, tools=None, tool_choice=None, parallel_tool_calls=False):
    """
    Simple get chat completion function
    """
   
    if tools:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls
    )
        return response.choices[0].message
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content