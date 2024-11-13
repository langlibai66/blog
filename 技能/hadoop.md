```
$sbin/start-dfs.sh 
$sbin/start-yarn.sh 
```





# hadoop学习报错解决

记录我在学习hadoop中遇到的报错信息

## 1、strat-dfs启动失败

输入指令 `sbin/start-dfs.sh`

发现无法启动 

报错

![image-20240923214239215](http://image.aaaieee.cn/image/image-20240923214239215.png)

苦苦学习，甚至因此没完成作业

最后终于找到问题所在，是因为权限问题导致，直接暴力将工作区文件的权限都改成777（不太安全，可以单独赋权）

后成功启动