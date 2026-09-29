import os


def calculator(a,b,op):
    if op=="+":
        return a+b
    elif op == "-":
        return a - b
    elif op == "*":
        return a * b
    elif op == "/":
        return a / b
    else:
        return "不支持的运算"



def text_length(text):
    return len(text)



def read_file(path):
    try:
        with open(path,"r",encoding="utf-8") as file:
            return file.read()

    except FileNotFoundError:
        return f"文件不存在:{path}"


def list_files(path="."):#如果调用这个函数的时候没有提供 path，默认就使用 "."。
    return os.listdir(path)