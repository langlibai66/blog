# MySQL

## 功能

- 实现对于大量数据的储存
- 增删改查速度更快
- 对于程序的对接比较友好

数据库的本质也是文件

- 数据库：用来管理数据表的
- 数据表：用来储存具体的数据的
- 字段：是用来储存一类信息的一列
- 记录：是用来描述一个人或事物的详细信息的一行
- 主键：用来表示唯一一行记录的特殊字段（数值存在且唯一）
- 关系型数据库：表和表之间可以发生关联的数据库

## 基础语法

- **DDL: 数据定义语言，用来定义数据库对象（数据库、表、字段）**
- **DML: 数据操作语言，用来对数据库表中的数据进行增删改**
- **DQL: 数据查询语言，用来查询数据库中表的记录**
- **DCL: 数据控制语言，用来创建数据库用户、控制数据库的控制权限**

### DDL(数据定义语言)

##### 数据库操作

```mysql
#查询所有数据库
SHOW DATABASES;
#查询当前数据库
SELECT DATABASE();
#创建数据库
CREATE DATABASE [ IF NOT EXISTS ] 数据库名 [ DEFAULT CHARSET 字符集] [COLLATE 排序规则 ];
#删除数据库
DROP DATABASE [ IF EXISTS ] 数据库名;
#使用数据库
USE 数据库名;
```

###### 注意：推荐字符集使用utf8mb4字符集

##### 表操作

```mysql
#查询当前数据库所有表：
SHOW TABLES;
#查询表结构：
#DESC 表名;
#查询指定表的建表语句：
SHOW CREATE TABLE 表名;
#创建表：
CREATE TABLE 表名(
    字段1 字段1类型 [COMMENT 字段1注释],
    字段2 字段2类型 [COMMENT 字段2注释],
    字段3 字段3类型 [COMMENT 字段3注释],
    ...
    字段n 字段n类型 [COMMENT 字段n注释]
)[ COMMENT 表注释 ];
#添加字段：
ALTER TABLE 表名 ADD 字段名 类型(长度) [COMMENT 注释] [约束];
#修改数据类型：
ALTER TABLE 表名 MODIFY 字段名 新数据类型(长度);
#修改字段名和字段类型：
ALTER TABLE 表名 CHANGE 旧字段名 新字段名 类型(长度) [COMMENT 注释] [约束];
#删除字段：
ALTER TABLE 表名 DROP 字段名;
#修改表名：
ALTER TABLE 表名 RENAME TO 新表名
#删除表：
DROP TABLE [IF EXISTS] 表名;
#删除表，并重新创建该表：
TRUNCATE TABLE 表名;
```

### DML(数据操作语言)

```mysql
#向指定字段添加数据：
INSERT INTO 表名 (字段名1, 字段名2, ...) VALUES (值1, 值2, ...);
#向全部字段添加数据：
INSERT INTO 表名 VALUES (值1, 值2, ...);
#批量添加数据：
INSERT INTO 表名 (字段名1, 字段名2, ...) VALUES (值1, 值2, ...), (值1, 值2, ...), (值1, 值2, ...);
INSERT INTO 表名 VALUES (值1, 值2, ...), (值1, 值2, ...), (值1, 值2, ...);
#修改数据：
UPDATE 表名 SET 字段名1 = 值1, 字段名2 = 值2, ... [ WHERE 条件 ];
#删除数据：
DELETE FROM 表名 [ WHERE 条件 ];
```

### DQL(数据查询语言)

#### 语法

```mysql
SELECT
    字段列表
FROM
    表名字段
WHERE
    条件列表
GROUP BY
    分组字段列表
HAVING
    分组后的条件列表
ORDER BY
    排序字段列表
LIMIT
    分页参数
```

```mysql
#查询多个字段：
SELECT 字段1, 字段2, 字段3, ... FROM 表名;
SELECT * FROM 表名;
#设置别名：
SELECT 字段1 [ AS 别名1 ], 字段2 [ AS 别名2 ], 字段3 [ AS 别名3 ], ... FROM 表名;
SELECT 字段1 [ 别名1 ], 字段2 [ 别名2 ], 字段3 [ 别名3 ], ... FROM 表名;
#去除重复记录：
SELECT DISTINCT 字段列表 FROM 表名;
```

特殊查询

```mysql
#条件查询
SELECT 字段列表 FROM 表名 WHERE 条件列表;
#聚合查询
SELECT 聚合函数(字段列表) FROM 表名;
#分组查询
SELECT 字段列表 FROM 表名 [ WHERE 条件 ] GROUP BY 分组字段名 [ HAVING 分组后的过滤条件 ];
#排序查询
SELECT 字段列表 FROM 表名 ORDER BY 字段1 排序方式1, 字段2 排序方式2;
#分页查询
SELECT 字段列表 FROM 表名 LIMIT 起始索引, 查询记录数;
```

###### 聚合函数<br/>

| 函数  | 功能     |
| ----- | -------- |
| count | 统计数量 |
| max   | 最大值   |
| min   | 最小值   |
| avg   | 平均值   |
| sum   | 求和     |

排序方式

- ASC: 升序（默认）
- DESC: 降序

##### 注意事项

- 起始索引从0开始，起始索引 = （查询页码 - 1） * 每页显示记录数
- 分页查询是数据库的方言，不同数据库有不同实现，MySQL是LIMIT
- 如果查询的是第一页数据，起始索引可以省略，直接简写 LIMIT 10

##### DQL执行顺序

FROM -> WHERE -> GROUP BY -> SELECT -> ORDER BY -> LIMIT

### DCL(数据控制语言)

```mysql
#查询用户：
USE mysql;
SELECT * FROM user;
#创建用户:
CREATE USER '用户名'@'主机名' IDENTIFIED BY '密码';
#修改用户密码：
ALTER USER '用户名'@'主机名' IDENTIFIED WITH mysql_native_password BY '新密码';
#删除用户：
DROP USER '用户名'@'主机名';
#查询权限：
SHOW GRANTS FOR '用户名'@'主机名';
#授予权限：
GRANT 权限列表 ON 数据库名.表名 TO '用户名'@'主机名';
#撤销权限：
REVOKE 权限列表 ON 数据库名.表名 FROM '用户名'@'主机名';
```

常用权限：

| 权限                | 说明               |
| :------------------ | :----------------- |
| ALL, ALL PRIVILEGES | 所有权限           |
| SELECT              | 查询数据           |
| INSERT              | 插入数据           |
| UPDATE              | 修改数据           |
| DELETE              | 删除数据           |
| ALTER               | 修改表             |
| DROP                | 删除数据库/表/视图 |
| CREATE              | 创建数据库/表      |

