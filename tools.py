import math
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


def search_file(filename,root="."):
    for current_path,dirs,files in os.walk(root):
        if filename in files:
            return os.path.join(current_path,filename)
    return f"未找到文件:{filename}"



def split_text(text,chunk_size=1000,overlap=100):
    chunks=[]
    step=chunk_size-overlap
    for i in range(0,len(text),step):
        chunk=text[i:i+chunk_size]
        chunks.append(chunk)

    return chunks


def search_chunks(chunks,keyword):
    results=[]

    for chunk in chunks:
        if keyword in chunk:
            results.append(chunk)

    return results

def cosine_similarity(a,b):
    dot_product = 0
    a_length=0
    b_length=0

    for x,y in zip(a,b):
        dot_product += x*y
        a_length+=x*x
        b_length+=y*y

    return dot_product/((a_length**0.5)*(b_length**0.5))