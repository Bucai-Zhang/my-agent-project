from tools import calculator


while True:
    user_input =input("你: ")

    if user_input == "退出":
        print("Agent:再见")
        break
    elif user_input =="计算":
        a=float(input("第一个数字："))
        op=input("运算符：")
        b = float(input("第二个数字："))

        result=calculator(a,b,op)
        print("Agent:",result)

    else:
        print("Agent:我收到你的任务了："+user_input)
