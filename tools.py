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


#如果调用这个函数的时候没有提供 path，默认就使用 "."。
def list_files(path="."):
    return os.listdir(path)


#寻找文件，.表示在当前目录下寻找，os.walk(root)表示读取当前目录数据，分别把地址，文件夹名字，文件名字传给三个参数，然后逐个进入文件夹寻找目标文件
def search_file(filename,root="."):
    for current_path,dirs,files in os.walk(root):
        if filename in files:
            return os.path.join(current_path,filename)
    return f"未找到文件:{filename}"


#划分语句块，重叠设置为overlap，增加容错
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

def semantic_search(chunks,chunk_vectors,query,model,top_k=3):
    query_vector=model.encode(query)

    results=[]

    for chunk,chunk_vector in zip(chunks,chunk_vectors):
        score = cosine_similarity(query_vector,chunk_vector)
        """
        # 这里：
        # ("第一块文本", 0.42)
        #
        # 叫一个
        # tuple（元组）
        """


        results.append((chunk,score))

    # 根据每个元素里的第
    # 1
    # 项，也就是
    # score，从大到小排序。
    #
    # 因为：
    # item[0] → chunkitem[1] → score
    #
    # reverse = True
    # 就是从大到小。
    # lambda 参数: 返回值

    results.sort(key=lambda item:item[1],reverse=True)
    return results[:top_k]
