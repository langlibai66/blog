#include <stdio.h>
#include <stdlib.h>
#include <string.h>
struct student
{
	char id[32];
	char name[32];
	float chinese_score;
	float math_score;
	float english_score;
}stu[10];
int main() 
{
	char file_path[100];
	char stu_inf[10][256];
	printf("请输入文件student.rec路径");
	scanf("%s",file_path);
	fflush(stdin);
	printf("请输入10个学生的信息：\n");
	for(int i = 0;i < 10;i++)
	{
		printf("请输入第%d个学生的信息\n",i+1);
		gets(stu_inf[i]);	
	}
	FILE*fp=fopen(file_path,"a");
	for(int i=0;i<10;i++)
	{
	  	sscanf(stu_inf[i],"%s%s%f%f%f",&stu[i].id,&stu[i].name,&stu[i].chinese_score,&stu[i].math_score,&stu[i].english_score);
	  	fprintf(fp,"学号：%s; 姓名：%s; 语文：%f; 数学：%f; 英语：%f\n",stu[i].id,stu[i].name,stu[i].chinese_score,stu[i].math_score,stu[i].english_score);
		fflush(stdin);
	}   
	fclose(fp);
	fp=fopen(file_path,"r");
	for(int i=0;i<10;i++)
	{
		char buf[100];
		fgets(buf,99,fp);
		printf("第%d个学生的信息为\n",i+1);
		printf("%s",buf);
		if(i<9)
		{
			printf(";");
		}
		
	}
	fclose(fp);
	return 0;
}

