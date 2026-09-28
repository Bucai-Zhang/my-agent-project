import json

from openai import OpenAI

from config import API_KEY, BASE_URL
from tools import calculator,text_length
from tool_schema import tools

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def run_agent(user_input):
    history=[
        {
            "role": "user",
            "content": user_input
        }
    ]

    while True:

        tool_called=False

        response = client.responses.create(
          model="deepseek-flash",
          input=history,
          tools=tools
        )

        history.extend(response.output)

        for item in response.output:
          if item.type == "function_call":
              tool_called=True

              arguments = json.loads(item.arguments)
              if item.name == "calculator":
                  result = calculator(
                    arguments["a"],
                    arguments["b"],
                    arguments["op"]
                  )

              elif item.name == "text_length":
                  result=text_length(
                      arguments["text"]
                  )

              history.append(
                  {
                    "type": "function_call_output",
                    "call_id": item.call_id,  #告诉模型是哪次工具调用的结果
                    "output": str(result)
                  }
              )
        if tool_called:
            continue
        return response.output_text



