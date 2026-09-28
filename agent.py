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

memory_summary=""
summarized_count=0





def run_agent(user_input):
    global memory_summary, summarized_count

    cutoff=len(conversation_history)-6
    if cutoff>summarized_count:
        new_old_history=conversation_history[summarized_count:cutoff]

        summary_response=client.responses.create(
            model="deepseek-flash",
            input=[
                {
                    "role":"user",
                    "content": f"""
                     这是之前已有的对话摘要：
                     {memory_summary}

                     这是新变成旧记录的对话：
                     {new_old_history}

                     请把两部分合并成一个简洁的历史摘要，保留以后对话可能有用的重要信息。
"""
                }
            ]
        )
        memory_summary=summary_response.output_text
        summarized_count=cutoff

        print("当前摘要：", memory_summary)
        print("已总结消息数：", summarized_count)


    working_history=[]
    if memory_summary:

        working_history.append(
           {
            "role": "system",
            "content": f"以下是之前对话的重要历史摘要：\n{memory_summary}"
           }
        )
    working_history.extend(
        conversation_history[-6:]
    )

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


