print("输入:",end='')
str=input()
sum=0
count=0
a=str.split()
for i in a:
    for n in i:
        count+=1
    for n in i:
        sum+=int(n)*2**(count-1)
        count-=1
print("输出:%d"%sum)        