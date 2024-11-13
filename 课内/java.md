# java

## jdk jre jvm

![image-20240602115409182](http://image.aaaieee.cn/image/image-20240602115409182.png)

![image-20240602115423724](http://image.aaaieee.cn/image/image-20240602115423724.png)

.java-------->.class----jvm---->机器语言

编写源文件 编译源文件生成字节码 加载运行字节码

java语句执行顺序  顺序 选择 循环 异常处理



## 基本语法

### 方法格式

```
权限修饰符 返回值声明 方法名称(参数列表){

		方法中封装的逻辑功能;

		return 返回值;

}
```

--权限修饰符

![image-20240602115432715](http://image.aaaieee.cn/image/image-20240602115432715.png)

--注释

```
//单行注释   

/*
多行注释
*/ 

/**
文档注释
**/
```

标识符举例

![image-20240602115441236](http://image.aaaieee.cn/image/image-20240602115441236.png)



## java变量

java是一个强类型语言 必须先声明类型后使用

**java数据类型分两大类 基本数据类型与引用类型**

![image-20240429212539612](http://image.aaaieee.cn/image/image-20240429212539612.png)

**引用数据类型：**

string 

数组 

接口 

类

按照声明位置进行定义分为局部变量与成员变量

### 变量的类型转换

boolean类型不参与转换

**自动类型转换**

容量小的类型自动转换成容量大的类型

byte,short,int -> float -> long ->double

byte short int之间不会互相转换 三者计算时会转化成int类型

**强制类型转换**

容量大的类型转换成容量小的类型时需要加上强制转换符

### 变量的作用域

在类体内定义的变量称为成员变量 作用域是整个类

在一个方法或方法内代码块中定义的变量称为局部变量

### 常量

量前加一个final





变量赋值注意事项：

```
float a = 133f
long a = 22220202l
char c = '羊'
```

## 数组

数组初始化方式

不允许在前面的[]里写元素个数

**动态两种**

```java
int[][] arr = new int[3][];
arr[0] = new int[3]

int [][] arr2 = new int[3][2]
arr[0][0] = 33
```

**静态一种**

```java
int arr4[][] = new int[][]{{1,2,3},{2,3,4}}
```

```java
arr.length 得到数组长度
```

```
 
```

![image-20240504154600252](http://image.aaaieee.cn/image/image-20240504154600252.png)

## 输入输出

scanner类型

```java
#输入
Scanner s = new Scanner(System.in);
s.nextInt();
s.nextLine();
s.nextfloat();
scanner.next();
#输出
System.out.println("XX");
```

 system.out.	  print() 普通输出

​							printf()格式化输出	

​							println()换行输出

## 类与对象

封装继承多态

我们进行一次举例

```java
public class Student {
	private String username;
	
	public String getUsername{
		return username;
	}
	
    #这个函数存在而不使用直接赋值的意义就是因为username这个变量是私有的
	public void setUsername(String username){
		this.username = username;
	}
}

class Test {
    public static void main(String[] args) {
        Student student=new Student();
        student.setUsername("张三");
        student.getUsername();
		System.out.println();
    }
}
```

类的实例化通过new语句进行创建

类的定义格式

```python
[修饰符] class 类名 [extends 父类名]  [implements 接口名]{
	//类体 包括类的成员变量与成员方法
}
```

## 继承

### 基类object

没有选择继承的时候默认继承object，有很多自带方法

### 继承格式

```java
public class Parent {
	private int age;
	
	public int getAge() {
		return age
	}
	
	public void setAge(int age) {
		this.age = age;
	}
	
	#有参
    public Parent(int age){
		this.age = age;
	}
	#无参
	public Parent(){
	}
	
	public void myprint(){
		system.out.println("我是父类的myprint方法");
	}
}

class Son extents Parent{
    public static void main(String[]args) {
        Son son = new son();
        son.age = 3;
    }
}
```

### 类的重写

对相同的函数进行再次声明就可以进行重写

## 类的封装

将类的某些信息隐藏在内部，不允许直接访问而是提供get set方法

```java
public class Person {
	private intn age;
	private string name;
	public String getName(){
		return name;
	}
	public int getAge(){
		return age;
	}
	public void setName(String name){
		this.name = name;
	}
	public void setAge(int age){
		this.age = age; 
	}
}
```

## 构造方法   重点

构造方法定义

主要用来创建对象时 初始化对象的 

总与new一起使用在创建对象的运算符中 

一个类可以有 多个构造函数 可根据参数个数不同或者参数类型不同区分 即构造函数的重载

 ![image-20240504184116949](http://image.aaaieee.cn/image/image-20240504184116949.png)

### 方法的重载重写

#### 重载

 ![image-20240504184435692](https://gitee.com/ai-yang-chenxu/img/raw/master/img/image-20240504184435692.png)

#### 重写

![](http://image.aaaieee.cn/image/image-20240504184541669.png)

区别

![image-20240504184920698](http://image.aaaieee.cn/image/image-20240504184920698.png)

### this关键字

在构造方法中指该构造器所创建的新对象

也就是对应对象的属性

也可以使用this关键字调出对象本身

例如在一个对象的setAge中调用getAge

**注意**：this只能在类的非静态方法中使用 静态方法与静态的代码块中不能出现this  原因 static方法在类加载时就已经存在了 但是对象在创建时才在内存中生成

### super关键字

super关键字主要存在于子类方法中

用于子类调用父类的方法

例如子类重写了父类的一个方法 但是又想重新调用一次父类的方法就使用super关键字

### static关键字

静态 的关键字

静态变量

静态方法

静态代码块

使用了static后的方法变成类方法 不需要new就能直接调用

### final关键字

final修饰的类不能被继承

final修饰的方法不能被重写 但是可以直接用

final修饰的基本类型变量不可变 但是引用类型变量引用不可改变 但是引用对象的内容可以改变

## 抽象类

在class前加一个abstract来修饰

抽象方法要在子类里进行实现 不然不正确

## 接口

将class替换为interface即可

接口里所有定义的方法实际上都是抽象的public abstract

变量只能为public static final类型的

public abstract void add();       **等效于**   void add();

## 抽象类与接口的区别

1. 接口要被子类实现 抽象类要被子类继承
2. 接口中变量全为公共静态常量 抽象类中可有普通变量
3. 接口中全为方法的声明  抽象类中可以有方法的实现
4. 接口中不可以有构造函数  抽象类中可以有构造函数
5. 接口可多实现 而抽象类必须被单继承
6. 接口中方法全为抽象方法 而抽象类中可以有非抽象方法

## 内存机制

### 栈

存放局部变量 不可以被多个线程共享

系统自动分配

空间连续 速度快

### 堆

存放对象  可以被多个线程共享

每个对象都有锁

空间不连续 速度慢 灵活

### 方法区

存放类的信息：代码 静态变量 字符串 常量等

可以被多个线程共享

空间不连续 速度慢 灵活

### 垃圾回收机制

程序员不能调用垃圾回收器 但是可以通过`system.gc()`建议回收

未引用的会被回收

finallize方法  每个对象都有这个方法 用来释放对象区域资源 一般不去调用

## 递归算法

递归头 什么时候不调用自己

递归体 什么时候调用自己

## 异常机制：

try catch finally          catch的顺序 **先小后大**

声明抛出异常：throws

手动抛出异常：throw

自定义异常：	首先继承Exception 或者它的子类

### 容器：

Collection接口： List -》ArrayList  LinkedList Vector

​							Set-》HashSet 内部使用HashMap实现



Map接口： 采用 key  value存储数据

​					HashMap线程不安全 效率高

​					HashTable线程安全 效率低



Iterator接口：遍历容器中元素



泛型：



Collections： 包含排序查找的工具类

字符串比较中 == 与 equal的区别

- ==：比较的是两个字符串内存地址（堆内存）的数值是否相等，属于数值比较；
-  equals()：比较的是两个字符串的内容，属于内容比较。



##  多态

多态体现为一个事物的多种形态 例如 父类引用变量可以指向子类对象

isinstanceof	

**向上转型** 将子类对象赋值给父类变量 

**向下转型** 将父类对象赋值给子类变量

## 注解

也叫元数据 用于描述数据的数据

基本注解：

@Override   重写 在重写的方法前加入即可

@SuppressWarnings  压制警告 在警告内容前加入 可以让我们暂时忽略特定的警告



自定义注解

```java
[public] @interface 注解名
{
数据类型 成员变量名（）[default 初始值]
}
```

注解跟类一样 会被编译为 注解名.class的字节码文件

成员变量名后面的（）必不可少

## 反射机制

一段程序在运行过程中 接受一个对象作为形参 该对象的编译时类型与运行时类型不一致 但是程序又需要调用该对象运行时的类中的方法

这就需要引用反射机制 保证在程序运行过程中

可以知道任意对象的运行时类型

可以构造任意类的对象

可以调用任意对象的属性和方法

**其实就是**在运行时获取对象的属性与方法，例如对象.getClass

## 内部类

将一个类作为成员放在另一个类或者方法的内部

嵌套类

内部类可以分为 非静态内部类和静态内部类

非静态内部类 是指 在非静态类的方法内访问某个变量时 先找局部变量 再找内部类的属性 最后找外部类的属性

如果局部变量 内部类属性 外部类三者名字相同

静态内部类是用static修饰的内部类都称为静态内部类

静态内部类是一个普通类 可以包含静态成员 也可以包含非静态成员

静态内部类不能访问外部类的实例成员 只能访问外部类的类成员

## lambda表达式

当接口中只有一个抽象方法 匿名内部类的语法过于频繁

这种接口叫做函数式接口

表达式 ： （形参列表）->{代码块}

形参列表：如果形参列表中只有一个参数 形参列表的圆括号也可以忽略

## 异常处理

### 基本语法

```java
try{
	执行语句
}catch (ExceptionType e) {
	异常处理
}finally{
	无论是否发生异常都会执行的语句
}
```

### 创建Exception

通过继承Exception来创建异常

```java
public class CustomException extends Exception{
	public CustomException(String message){
		super(message)
	}
}
```

### throw/throws

用于手动抛出异常	需要使用

```java
public void processAge(int age) {
    if (age < 0) {
        throw new IllegalArgumentException("Age cannot be negative");
    }
    // 其他处理逻辑
}
```

**throws**

```java
public String readFile(String fileName) throws IOException {
    // 读取文件内容的逻辑
}
```

## 输入输出操作

### InputStream

`InputStream`是用于从各种源（如文件、网络连接等）读取字节流的抽象类。它定义了一系列用于读取字节的方法。你可以使用`InputStream`来读取二进制数据，比如图片、音频或视频文件。

`OutputStream`是用于向各种目标（如文件、网络连接等）写入字节流的抽象类。它定义了一系列用于写入字节的方法。你可以使用`OutputStream`来写入二进制数据，比如将数据写入文件或通过网络发送。

`Reader`是用于从各种源（如文件、网络连接等）读取字符流的抽象类。它定义了一系列用于读取字符的方法。你可以使用`Reader`来读取文本数据，比如读取文本文件中的内容。

`Writer`是用于向各种目标（如文件、网络连接等）写入字符流的抽象类。它定义了一系列用于写入字符的方法。你可以使用`Writer`来写入文本数据，比如将数据写入文本文件。

```java
// 使用FileReader读取文件
FileReader fileReader = new FileReader("file.txt");
int data = fileReader.read(); // 读取一个字符
while (data != -1) {
    System.out.print((char)data);
    data = fileReader.read();
}
fileReader.close();

// 使用FileWriter写入文件
FileWriter fileWriter = new FileWriter("file.txt");
fileWriter.write("Hello, world!");
fileWriter.close();
```

### System.in、System.out 和 System.err

`System.in`、`System.out`和`System.err`是Java中的三个标准I/O流。

- **System.in**：标准输入流，通常对应于键盘输入。你可以使用它来从控制台读取用户的输入。
- **System.out**：标准输出流，通常对应于控制台输出。你可以使用它向控制台输出信息。
- **System.err**：标准错误流，也通常对应于控制台输出。与`System.out`不同的是，它主要用于输出错误信息。

## 泛型

**java 中泛型标记符：**

- **E** - Element (在集合中使用，因为集合中存放的是元素)
- **T** - Type（Java 类）
- **K** - Key（键）
- **V** - Value（值）
- **N** - Number（数值类型）
- **？** - 表示不确定的 java 类型

### Collection \<E>

Collection 是 Java 集合框架中所有集合类的根接口。它代表了一组对象，这些对象通常称为元素。Collection 接口的主要特点包括：

- **存储一组对象**：Collection 是一个容器，可以存储多个对象，这些对象可以是任何类型，包括基本类型的封装类、自定义对象等。
- **无序性**：Collection 不保证元素的顺序，即它们不一定按照插入的顺序进行存储和访问。
- **允许重复元素**：Collection 允许存储重复的元素，即相同的对象可以被添加多次。
- **常见实现类**：Java 中常见的 Collection 实现类包括 List、Set 和 Queue 接口的各种实现类，如 ArrayList、LinkedList、HashSet 等。

### Map<K,V>

Map 接口代表了一种映射关系，它将键映射到值。Map 中的键是唯一的，而值则可以重复。Map 接口的主要特点包括：

- **键值对存储**：Map 存储的是键值对，每个键都映射到一个值。通过键可以快速查找对应的值。
- **键的唯一性**：Map 中的键是唯一的，每个键最多只能与一个值关联。
- **值的重复性**：Map 中的值可以重复，即不同的键可以映射到相同的值。
- **常见实现类**：Java 中常见的 Map 实现类包括 HashMap、TreeMap、LinkedHashMap 等。
