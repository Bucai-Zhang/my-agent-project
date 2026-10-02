import json

from openai import OpenAI

from config import API_KEY, BASE_URL
from tools import calculator,text_length,read_file,list_files,search_file
from tool_schema import tools

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

tool_map={
    "calculator":calculator,
    "text_length":text_length,
    "read_file":read_file,
    "list_files":list_files,
    "search_file":search_file
}
#加入系统工具使用策略
SYSTEM_PROMPT = """
你是一个可以调用工具完成任务的 Agent。

文件工具使用规则：
1. 如果用户已经提供明确文件路径，直接使用 read_file。
2. 如果用户知道文件名但不知道具体路径，优先使用 search_file。
3. search_file 已经成功返回路径后，直接使用 read_file，不需要再调用 list_files 验证。
4. list_files 主要用于用户要求查看某个目录中有哪些文件时。
"""



conversation_history=[]

memory_summary=""
summarized_count=0



def update_memory_summary():
    global memory_summary,summarized_count
    cutoff=len(conversation_history)-6
    if cutoff>summarized_count:
        new_old_history=conversation_history[summarized_count:cutoff]
        memory_response=client.responses.create(
            model="deepseek-flash",
            input=[{

                "role":"user",
                "content": f"""
                这是之前已有的对话摘要:
                {memory_summary}
                
                这是新变成记录的对话摘要:
                {new_old_history}
                
                请把两部分合并成一个简洁的历史摘要，保留以后对话可能有用的重要信息。
                """
            }
            ]
        )

        memory_summary=memory_response.output_text
        summarized_count=cutoff

        print("当前摘要：", memory_summary)
        print("已总结消息数：", summarized_count)



def build_working_history(user_input):

    working_history = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

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
    return working_history


def generate_rag_answer(query,context):
    response=client.responses.create(
        model="deepseek-flash",
        input=[
            {
                "role": "system",
                "content":"请根据提供的参考资料回答问题。如果资料中没有答案，就明确说资料中没有相关信息。"
            },
            {
                "role": "user",
                "content":f"""
                参考资料:{context}
                
                问题:{query}
             """
            }
        ]
    )
    return response.output_text





def execute_tool_call(item):
    print(f"当前调用工具：{item.name}")
    print(f"当前调用参数：{item.arguments}")
    arguments = json.loads(item.arguments)

    tool = tool_map.get(item.name)
    if tool is not None:
        result = tool(**arguments)
    else:
        result = f"未找到工具:{item.name}"
    return result



def run_agent(user_input):
    update_memory_summary()
    working_history = build_working_history(user_input)

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

                result=execute_tool_call(item)

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


