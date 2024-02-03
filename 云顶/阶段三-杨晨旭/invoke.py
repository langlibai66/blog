import AIhomework
if __name__=='__main__':
    print("作业1.二进制转化 2.构造最短回文字符串 3.AI成员信息查询")
    choice=int(input("请选择你想看的作业:"))
    if choice == 1:
        print("输入二进制数")
        AIhomework.binary()
    if choice == 2:
        print("输入字符串")
        AIhomework.minstr()
    if choice == 3:
        AIhomework.search()
