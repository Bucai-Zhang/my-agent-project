import os
import json
from openai import OpenAI

client=OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"

)


tools = [
    {
        "type":"function",
        "name":"calculator",
        "description":"用于进行两个数字的加减乘除运算",
        "parameters":
            {
                "type":"object",
                "properties":{
                    "a":{
                        "type":"number"
                    },
                    "b":{
                        "type":"number"
                    },
                    "op": {
                        "type": "string",
                        "enum": ["+", "-", "*", "/"] #限制大模型的输出，防止输出无法识别的字符
                    }
                },
                "required": ["a", "b", "op"]
            }
    }
]

from tools import calculator
user_input = input("你: ")
response = client.responses.create(
    model="deepseek-flash",
    input=user_input,
    tools=tools
)


for item in response.output:
    if item.type == "function_call":
        arguments=json.loads(item.arguments) #把JSON格式字符串变成Python字典格式
        result=calculator(
            arguments["a"],
            arguments["b"],
            arguments["op"]
        )
        print("工具执行结果",result)
        second_response = client.responses.create(
            model="deepseek-flash",

            input=[
                {
                   "role":"user",
                   "content":user_input
                },
                *response.output,
                {
                "type":"function_call_output",
                "call_id":item.call_id,
                "output":str(result)
                }
            ]
        )
        print(second_response.output_text)




""" 
工具箱 tools
[
     工具1 {...},
     工具2 {...},
     工具3 {...}
]
[] = list，多个东西的集合  列表

{} = dict，一个东西的属性集合  字典

"type": "object"
= JSON Schema 里说“这里应该是一个对象结构”
"""


"""response
│
├─ id = 整次响应的编号
│
└─ output = 输出项列表
    │
    ├─ item1
    │   └─ 可能有自己的 id
    │
    └─ item2
        ├─ id
        ├─ call_id
        ├─ name
        └─ arguments"""
