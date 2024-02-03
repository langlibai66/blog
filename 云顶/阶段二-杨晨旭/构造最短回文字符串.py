anwser=-1
user_input=input("输入:")
str_num=len(user_input)
for i in range(str_num-1):
   if user_input[-1]==user_input[i]:
      index=i
      right=-1
      left=index
      for ii in range((str_num-1-index)//2+1):
         left+=1
         right-=1
         if ii==((str_num-1-index)//2):
            if user_input[left]==user_input[right]:
               anwser=index
         elif user_input[left]==user_input[right]:
            continue
         else:
            break
   else:
      anwser=-1
   if	anwser!=-1:
      break
output=[]
new=[]
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
endput=[]
same=1
endput.append(output[0])
for num in range(1,len(output)):
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
