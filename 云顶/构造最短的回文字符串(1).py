s=input()
s1=list(s)
s1.reverse()
s2=list(s)
s3=[]
s4=[]
s6=[]
s5=2
k=0
num=1
if s1==s2:
    print("""添加的字符串为:"None",输出的结果为"""+''.join(s2))
else:
    n=len(s2)
    for i in range(len(s2)):
        if s2[i]!=s2[-i-1]:
            s2.insert(n,s2[i])#给字符串补上需要的元素
            s3.append(s2[i])#补上的元素是什么
        else:
            break
    print("添加的字符串为:"+''.join(s3))
    print("输出的结果是"+''.join(s2))
    for i in range(len(s2)-2):#判断连续相同元素的个数
        num=1
       # for k in range(len(s2)-1):
            #if k<=i:
             #   continue
        if s2[i]==s2[i+1]:
            s4.append(s2[i])
            while i<len(s2):
                num=+1
                k=i+1
                k=k+1
                if s2[i]!=s2[k]:
                    s4.append(num)
                    i=k
                    break
        if s2[i]!=s2[i+1]:
            s4.append(s2[i])      
    print(s4)
    #    s4.append(1)
'''    else:
        s4.append(0)
for i in range(len(s4)):
    if s4[i]==1:
        if s4[i+1]==1:
            s5=s5+1
            s6.append(s5)
        else:
            s5=s5
            s6.append(s5)
print("".join(str(x) for x in s4))
print("".join(str(x) for x in s6))'''
            #定义一个函数，每找到一个相同的连续元素，就输出他们的个数，并且输出最后一个的位置，然后，使用循环，最后把每个个数都打出来，
                #这个函数还能够把这个相同的元素输出，然后最终可以用一for循环，把所有的新的字符串拼接起来
