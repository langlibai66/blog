# 初识

## [1]简介

#### Beautiful Soup是Python的一个库，能够从网页抓取数据提供了一些函数用来处理数据，用很少的代码就能够写出来一个完整的程序。

## [2]特点

#### Beautiful Soup能够自动的将文档转换为utf-8编码格式，能够更为方便的进行使用

# 使用

## [1]创建对象

要想创建一个Beautiful Soup对象，首先要导入库bs4，lxml，requests。

这里使用一个实例

```python
html = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title" name="dromouse"><b>The Dormouse's story</b></p>
<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1"><!-- Elsie --></a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>
<p class="story">...</p>
"""
soup = BeautifulSoup(html,'lxml')  #创建 beautifulsoup 对象
```

同时也可以使用HTML文件直接创建对象

```python
soup1 = BeautifulSoup(open('index.html')) 
```

##### Beautiful Soup将复杂HTML文档转换成一个复杂的树形结构,每个节点都是Python对象,所有对象可以归纳为4种:

- Tag
- NavigableString
- BeautifulSoup
- Comment

#### 1>Tag

Tag就是 HTML 中的一个个标签

通过这种方法就可以直接访问标签

```python
print (soup.title)
print (soup.head)
print (soup.a)
print (soup.p)
print (type(soup.a))
```

Tag有两个重要的属性，name与attrs

并且可以对这些属性和内容进行修改

#### 2>**NavigableString**

得到了标签的内容用 .string 即可获取标签内部的文字

例如

```python
print(soup.a.string)
```

#### 3>BeautifulSoup

BeautifulSoup 对象表示的是一个文档的全部内容.大部分时候,可以把它当作 Tag 对象，是一个特殊的 Tag，我们可以分别获取它的类型，名称：

#### 4>**Comment**

## [2]遍历文档树

#### tag 的 .contents 属性可以将tag的子节点以列表的方式输出

#### **.children**属性生成的是一个列表生成器对象，可以通过遍历获得其中的内容

#### .descendants 属性可以对所有tag的子孙节点进行递归循环，和 children类似，要获取其中的内容，我们需要对其进行遍历

还有许多关联选择方法

## [3]CSS选择器

CSS的使用只需要调用select方法结合CSS选择器语法就可以实现元素定位

格式：soup.select()

1>id选择器

每段节点都有专属的id，通过id就可以匹配相应的节点，使用#加id就可以实现定位

2>类选择器

类选择器用.来定位元素

例如<p class="title" name="dromouse"><b>The Dormouse's story</b></p>中的title就是类名

3>标签选择器

直接使用标签来定位即可例如p

###### 在定位时可以多种定位方法混合使用，达到更为理想的效果。

## [4]CSS高级用法

#### 1>嵌套选择

因为其仍旧是一个tag对象，所以可以使用嵌套选择

#### 2>属性获取

使用attrs属性

#### 3>文本获取

使用string与strings获取文本信息

## [5]方法选择器

#### 1>find_all()方法

soup传入整个HTML文档然后进行全文搜索

find_all(name,attrs,recursive,text,**kwargs)

**name参数**就是查找名字为name的节点，返回tag对象，这个传入的参数可以是字符串，可以是列表，正则，还可以是True(以此为参数可以获取所有的标签)

**attrs参数**是查询含有接受的属性值的标签，参数形式为字典类型例如{id:'link1'}

**kwargs参数**是接收常用的属性参数的，如id和class，参数形式是变量赋值的形式，例如(id='link1')<br/>*class 是关键字，使用时应该加下划线即*class_=……

**text参数**是通过查询含有接受的文本的标签来查找的，参数形式是字符串

**limit参数**是用于限制返回结果的数量的参数，参数形式是整数，顾名思义

**recursive参数**是决定是否获取子孙节点的参数，参数形式是布尔值，默认是True

#### 2>find方法

返回值是第一个符合条件的元素，除limit参数不存在外，其余参数都与find_all方法一致