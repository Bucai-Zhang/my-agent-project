import json

from openai import OpenAI

from config import API_KEY, BASE_URL
from tools import calculator,text_length
from tool_schema import tools

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

tool_map={
    "calculator":calculator,
    "text_length":text_length
}

conversation_history=[]






def run_agent(user_input):
    working_history=conversation_history.copy()
    working_history.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    while True:

        tool_called=False

        response = client.responses.create(
          model="deepseek-flash",
          input=working_history,
          tools=tools
        )

        working_history.extend(response.output)

        for item in response.output:
          if item.type == "function_call":
              tool_called=True

              arguments = json.loads(item.arguments)


              tool=tool_map.get(item.name)
              if tool is not None:
                  result=tool(**arguments)
              else:
                  result=f"未找到工具:{item.name}"


              working_history.append(
                  {
                    "type": "function_call_output",
                    "call_id": item.call_id,  #告诉模型是哪次工具调用的结果
                    "output": str(result)
                  }
              )
        if tool_called:
            continue

        answer = response.output_text
        conversation_history.append(
            {
                "role": "user",
                "content": user_input
            }
        )
        conversation_history.append(
            {
                "role": "assistant",
                "content":answer
            }
        )
        return answer


