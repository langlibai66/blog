names = ['感知2201-何秉轩', '机电2205-谢伟', '智能2202-王智申', '区块链2201-王溪慈', '数科2201-李诗语', '光信息2204-周毅涵', '秘书处-英语2203-赵靖涵', '计科2202-李翼廷', '软件2241-张开然', '软件2209-刘思洲', '软件2242-王彤', '秘书处-电科2201-史翊坤', '数学2201-宋小娜', '软件2221-刘清湲', '暖通2202-王泽宇', '应急2201-赵文恺', '医工2201-张春慧', '秘书处-英语2203-刘怡杉', '软件2241-王鹏超', '软件2207-李喆', '数科2201-于翔川', '智能2202-王一帆', '软件2209-王玥', '高材2202-刘泽', '电气2204-郭时宇', '成型2207-陈新阳', '软件2231-张硕航', '秘书处-英俄2201-李若璇', '数科2202-赵婷婷', '软件2207-郭鸿凯', '物联网2202-行亚楠', '物联网2201-封鑫田', '智测2201-侯华润', '地质2102-郭明艺', '秘书处-物流2202-冯思雨', '数科2203-郝少敏', '软件2214-陈柔曼', '机械2205-李峥', '智能2202-吴彦兴', '物联网2201-张益德', '软件2244-牛玮茗', '软件2219-谢昊洋', '工程力学2202-刘嘉琪', '软件2239-冯雲', '软件2242-杨焰琴', '智能2201-杨晨旭', '软件2212-樊泓昱', '智能2202-王宇舟', '软件2243-任博轩', '数科2204-张雪涵', '智能2202-王君', '光信息2204-吴昊', '资勘2201-杨梅', '软件2220-张明磊', '数科2204-赵佳宝', '软件2218-刘荣鑫', '软件2203-李晓慧', '网安2202-王健炫', '安全2202-张一超', '智能2202-解金洋', '数科2204-郭淑娟', '智能2202-夏景琦', '安全2205-雷蒙阳']
genders = ['male', 'male', 'male', 'female', 'female', 'male', 'female', 'male', 'male', 'female', 'unknown', 'male', 'female', 'unknown', 'male', 'male', 'female', 'unknown', 'male', 'female', 'unknown', 'male', 'female', 'male', 'female', 'male', 'male', 'female', 'female', 'male', 'unknown', 'female', 'female', 'female', 'female', 'female', 'female', 'male', 'male', 'male', 'male', 'male', 'unknown', 'male', 'female', 'unknown', 'male', 'male', 'male', 'female', 'male', 'unknown', 'unknown', 'unknown', 'unknown', 'male', 'unknown', 'male', 'male', 'male', 'female', 'unknown', 'male']
dictionary={'感知2201':[]}
for num in range(len(names)):
    oneinf = names[num].split('-')
    if oneinf[0] == "秘书处":
        dictionary[oneinf[1]] = []
    else:
        dictionary[oneinf[0]] = []
for str in dictionary:
    inflist = []
    for num in range(len(names)):   
        inftuple = ()
        oneinf = names[num].split('-')
        if oneinf[0] == "秘书处" and oneinf[1]==str:
            inftuple = (oneinf[2])
            inflist.append(inftuple)
        elif oneinf[0] == str:
            inftuple = (oneinf[1],genders[num])
            inflist.append(inftuple)
    dictionary[str] = inflist
while 1:
    print(("1.根据班级或名称查询 2.查询人数最多班级及其性别比例 3.退出"))
    choice = int(input("请输入选择:"))
    if choice == 1:
        search = input("输入:")
        index=0
        count = 0
        output=[]
        for str in dictionary:
            if search == str:
                index=1
                for num in range(len(dictionary[str])):
                    output.append(dictionary[str][num][0])
                    count+=1
                continue
            for num in range(len(dictionary[str])):
                if search == dictionary[str][num][0]:
                    index=2
                    output.append(str)
        if index == 0:
            print("输出:您查找的信息不存在")
        if index == 1:
            print("输出:%s班有%d位同学%s"%(search,count,','.join(output)))
        if index == 2:
            print("输出:%s所在的班级为:%s"%(search,','.join(output)))
    if choice == 2:    
        num = 0
        for str in dictionary:
            inum = len(dictionary[str])
            if inum > num:
                num = inum
                max_class=str
        malenum=0
        femalenum=0
        unknownnum=0
        for num in range (len((dictionary[max_class]))):
                if dictionary[max_class][num][1] == 'male':
                    malenum += 1
                if dictionary[max_class][num][1] == 'female':
                    femalenum += 1
                if dictionary[max_class][num][1] == 'unknown':
                    unknownnum+=1
        sum=malenum+femalenum+unknownnum
        print("该班级男生比例为%d/%d,女生比例为%d/%d,未知性别比例为%d/%d"%(malenum,sum,femalenum,sum,unknownnum,sum))
    if choice == 3:
        break            






