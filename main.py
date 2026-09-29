from agent import run_agent
from tools import cosine_similarity


a=[1,0]
b=[0,1]

results = cosine_similarity(a,b)

print(results)


while True:


    user_input = input("你: ")

    if user_input == "退出":
        break

    answer = run_agent(user_input)

    print("Agent:", answer)
