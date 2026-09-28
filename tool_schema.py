# 模型来认识工具

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
    },

    {
        "type":"function",
        "name":"text_length",
        "description":"计算一段文本的字符长度",
        "parameters":
            {
                "type":"object",
                "properties":{
                    "text":{
                        "type":"string"
                    }
                },
                "required": ["text"]
            }
    }
]