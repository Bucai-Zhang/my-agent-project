import json

from openai import OpenAI

from config import API_KEY, BASE_URL
from tools import calculator
from tool_schema import tools

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def run_agent(user_input):
    response = client.responses.create(
        model="deepseek-flash",
        input=user_input,
        tools=tools
    )
    for item in response.output:
        if item.type == "function_call":
            arguments = json.loads(item.arguments)
            result = calculator(
                arguments["a"],
                arguments["b"],
                arguments["op"]
            )
            second_response = client.responses.create(
                model="deepseek-flash",
                input=[
                    {
                        "role": "user",
                        "content": user_input
                    },
                    *response.output,
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": str(result)
                    }
                ]
            )
            return second_response.output_text
    return response.output_text
