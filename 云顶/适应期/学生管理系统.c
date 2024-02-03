#include<stdio.h> 
#include<string.h>
typedef struct student {             //结构体建立 
	char number[32];
	char name[10];
	char sex[4];
	int age;
	char phone[64]; 
}student;
student stu[1024];
int  assignment(FILE*fp); 
int main()
{
	FILE*fp;
	fp=fopen("D://students.txt","r");
	int num=assignment(fp);
	printf("请输入");
	int choice=0;
	char key[64];
	int key0,key1,key2;
while(choice != -1){
	printf("请输入数字选择查询模式\n1：根据姓名或学号查询学生信息查询模式\n2：根据年龄范围查询学生信息 \n3：根据班级查询学生信息\n4：添加学生\n5: 退出查询\n");
	scanf("%d",&choice);
	switch(choice){
		case 1:
			printf("请输入你要查找的姓名或学号\n");
			printf("输入:");
			scanf("%s",&key);
			int i;
			for (i = 0;i < num;i++){
				if(strcmp(key,stu[i].name)==0){
					printf("%s\t%s\t%s\t%d\t%s\n",stu[i].number,stu[i].name,stu[i].sex,stu[i].age,stu[i].phone);
				}
				if(strcmp(key,stu[i].number)==0){
					printf("%s\t%s\t%s\t%d\t%s\n",stu[i].number,stu[i].name,stu[i].sex,stu[i].age,stu[i].phone);				}
			}
			break;
		case 2:
			printf("请输入你要查找的年龄范围 (最大值与最小值)\n示例 19 20\n输入:");
			scanf("%d %d",&key1,&key2);
			for (i = 0;i < num;i++){
				int key0=stu[i].age;
				if(key1 <=key0 &&key0<=key2){
					printf("%s\t%s\t%s\t%d\t%s\n",stu[i].number,stu[i].name,stu[i].sex,stu[i].age,stu[i].phone);				}
			}
		break; 
		case 3:
			printf("请输入你要查找的班级\n");
			printf("输入:");
			char key3[64];
			scanf("%s",key3);
			for (i = 0;i < num;i++){
				if(strcmp(key3,stu[i].phone) == 0){
					printf("%s\t%s\t%s\t%d\t%s\n",stu[i].number,stu[i].name,stu[i].sex,stu[i].age,stu[i].phone);
				}
			}
			break;
		case 4:
			 fp=fopen("D://students.txt","w");
						 
		case 5:
			choice=-1; 
			break;
	}
}
	fclose(fp);
	return 0;
}
int  assignment(FILE*fp)
{
	fp=fopen("D://students.txt","r");
	char buf[128];
	int n=0;
	if(fgets(buf,sizeof(buf),fp) == NULL){
		printf("未读取到文件");
	}
	rewind(fp);   
	while(fgets(buf,sizeof(buf),fp) != NULL){
	
		sscanf(buf,"%[^,]",stu[n].number);
		sscanf(buf,"%*[^,],%[^,]",stu[n].name);
		sscanf(buf,"%*[^,],%*[^,],%[^,]",stu[n].sex);
		sscanf(buf,"%*[^,],%*[^,],%*[^,],%d",&stu[n].age);
		sscanf(buf,"%*[^,],%*[^,],%*[^,],%*d,%[^\n]\n",stu[n].phone);
		n++;
	}
	return n;
}

