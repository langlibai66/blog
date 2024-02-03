#include<stdio.h>
struct student
{
	char a[20];
	char b[5];
	int c;
	int d;
	int e;
};
int main()
{
	struct student grade[15];
	printf("请输入15名学生的学号，姓名，3门课的成绩（以空格分隔，以回车结束一行的数据输入）：\n");
	int i;
	for(i=0; i<15; i++)
	{
		scanf("%s %s %d %d %d\n", &grade[i].a, &grade[i].b, &grade[i].c, &grade[i].d, &grade[i].e);	
	}
	
	int max=(grade[0].c + grade[0].d + grade[0].e);
	int sum;
	int ret=0;
	for(i=1; i<15; i++)
	{
		sum = (grade[i].c + grade[i].d + grade[i].e);
		if(sum>max)
		{
			max=sum;
			ret=i;
		}
	}
	printf("最高分学生的数据是->学号:%s\n姓名:%s\n3门课程成绩:%d %d %d\n", grade[ret].a, grade[ret].b, grade[ret].c, grade[ret].d, grade[ret].e);
	return 0;

}
