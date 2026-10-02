from rag import rag_query
from sentence_transformers import  SentenceTransformer






while True:


    query = input("你: ")

    if query == "退出":
        break

    answer=rag_query(query)


    print("Agent:", answer)
