from tools import calculator


while True:
    user_input =input("你: ")

    if user_input == "退出":
        print("Agent:再见")
        break
    elif "计算" in user_input:
        expression=user_input.replace("帮我计算","").strip()
        for op in["+","-","*"]:
            if op in expression:
                a_text,b_text=expression.split(op)
                a=float(a_text)
                b =float(b_text)

                result=calculator(a,b,op)
                print("Agent:",result)
                break

    else:
        print("Agent:我收到你的任务了："+user_input)
