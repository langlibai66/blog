user_input=input("输入:")
output=[]
endput=[]
new=[]
anwser=-1
same=1
str_num=len(user_input)
for i in range(str_num-1):
    if user_input[-1]==user_input[i]:
        index=i
        for ii in range((str_num-1-index)//2+1):
            left=index+ii
            right=str_num-1-ii
            if ii==((str_num-1-index)//2):
                if user_input[left]==user_input[right]:
                    anwser=index
            elif user_input[left]==user_input[right]:
                continue
            else:
                break
    if anwser!=-1:
        break
if anwser==-1:
    anwser=str_num-1
if anwser==-2:
    for char in user_input:
        output.append(char)
else:      
    for char in user_input:
        output.append(char)
    index=anwser-1
    while index>=0:
        output.append(user_input[index])
        new.append(user_input[index])
        index-=1
for num in range(len(output)):
    if output[num]==output[num-1]:
        same+=1
        if num == len(output)-1:
            if same!=1:
                endput.append(str(same))
    else:
        if same!=1:
            endput.append(str(same))
        endput.append(output[num])
        same=1
print("添加的字符串为:%s"''.join(new),end="")
print(",输出的结果为:%s"%''.join(endput))
