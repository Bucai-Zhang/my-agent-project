# 模型来认识工具

tools = [
    {
        "type":"function",
        "name":"calculator",
        "description":"用于进行两个数字的加减乘除运算",
        "parameters":
            {
                "type":"object",#模型参数必须按照JSON object的格式输出
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
    },

    {
        "type": "function",
        "name": "read_file",
        "description": "读取指定文本文件内容",
        "parameters":
            {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string"
                    }
                },
                "required": ["path"]
            }
    },

    {
        "type": "function",
        "name": "list_files",
        "description": "列出指定文件夹中的文件和子文件夹",
        "parameters":
            {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string"
                    }
                },
                "required": ["path"]
            }
    }
]



"""parameters
│
├── type = object
│   → 整体参数必须是一个 JSON 对象
│
├── properties
│   └── path
│       └── type = string
│       → 这个对象里面允许有 path，
│         path 的值必须是字符串
│
└── required = ["path"]
    → path 不能缺"""