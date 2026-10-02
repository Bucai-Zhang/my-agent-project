from sentence_transformers import SentenceTransformer
from tools import read_file,split_text,semantic_search
from agent import generate_rag_answer



model=SentenceTransformer("BAAI/bge-small-zh-v1.5")

text=read_file("test.txt")

chunks = split_text(text,120,30)

chunk_vectors=model.encode(chunks)

threshold=0.4

def rag_query(query):

    top_results=semantic_search(
        chunks,
        chunk_vectors,
        query,
        model,
        top_k=3
    )

    filtered_chunks=[]

    for chunk,score in top_results:
        if score>threshold:
            filtered_chunks.append(chunk)

    if len(filtered_chunks)==0:
        return "参考资料中没有找到足够相关的信息。"
    # join
    # 的作用是：
    # 把很多字符串连接成一个大字符串。比如：texts = ["苹果", "香蕉", "西瓜"]
    # 如果result = "-".join(texts)得到苹果-香蕉-西瓜
    context="\n\n".join(filtered_chunks)

    answer=generate_rag_answer(query,context)

    return answer
