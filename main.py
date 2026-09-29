from agent import run_agent



while True:

    user_input = input("你: ")

    if user_input == "退出":
        break

    answer = run_agent(user_input)

    print("Agent:", answer)
