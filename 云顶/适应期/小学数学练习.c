#include<stdio.h>
#include<stdlib.h>
#include<time.h>
int land(char[]);
int question();
int goal=-1;
clock_t start_time, end_time;
char inf[10];
FILE * fp;
int main()
{
	srand((unsigned int)time(NULL));
	land(inf);
	int choice=1;
	char q[10];
	while(choice<4&&choice>0)
	{ 
	printf("ģʽ��(1)��ʼ���� (2)������ (3)�˳�\n����������ѡ��ģʽ");
	scanf("%d",&choice);
	switch(choice)
	{
		case 1:
			fp=fopen("D://record.txt","a");
			printf("���ǽ�����10����ѧ�⣬������������\n");
			goal++;
			start_time = clock();
			goal = question();
			end_time = clock(); 
			int time = (int)((end_time - start_time)/CLOCKS_PER_SEC);
			printf("���κ�ʱ%d��",time);
			fprintf(fp,"ID:%s %d�� %d�� ",inf,goal*10,time);
			fclose(fp);
			break;
		case 2:
			if(goal==-1)
			{
				printf("����û�н��й����⣬����д�����ڽ��д˲���");
			}
			else{
				printf("���ĵ÷���%d\n",10*goal);
			}
			break;
		case 3:
			choice=-1;
			break;
	}
	}
} 
int land(char inf[10]) 
{
	int flag=1; 
	while(flag==1){
		printf("���������ID\n");
		gets(inf);
		char a;
		char b;
		int flag=0,c=-1,d=-1,e=-1,f=-1;
		sscanf(inf,"%c%c%1d%1d%1d%1d",&a,&b,&c,&d,&e,&f);
		printf("%d",c);
		if((!(a>='A'&&a<='Z'))||(!(b>='A'&&b<='Z'))||(c>9||c<0)||(d>9||d<0)||(e>9||e<0)||(f>9||f<0))
		{
			flag=1; 
			printf("�������ID����ȷ\n");
		}
		else{
			printf("��½�ɹ�\n");
			break;
		}
	}
	return 0;
}
int question()
{
	int point = 0;
	int n[10];
	int myan[10];
	int usenum[4]={0,0,0,0};
	char usesign[10]={};
	char sign[4]={'+','-','*','/'};
	int special,q,e;
	int anwser[10];
	int flag=0;
	int i;
	int qq[10];
	int ee[10];
	do
	{
		flag=0;
		for (i=0;i<10;i++)
		{
			special = rand()%4;
			usesign[i]=sign[special];
			usenum[i]++;
		}
		for(i=0;i<4;i++)
		{
			if(usenum[i]==0)
			{
				flag=1;
				break;
			}
		}
		for (i=0;i<9;i++)
		{
			if(usesign[i]==usesign[i+1]){
				flag = 1;
				break;
			}
		}
	}while(flag==1);
		for(i=0;i<10;i++){
			do{
			flag=0;
			q=rand()%99+1;
			e=rand()%99+1;
				if('*'==usesign[i]){
					if((q*e)>=100||(q*e)<=0){
						flag=1;
						continue;
					}
					else{
						anwser[i]=q*e;
						qq[i]=q;
						ee[i]=e;
					}
				}
				else if(usesign[i]=='/'){
					if((q/e) >= 100||(q/e) <= 0||(q%e) != 0){
						flag = 1;
						continue;
					}
					else{
						anwser[i]=q/e;
						qq[i]=q;
						ee[i]=e;
	
					}
				}
				else if(usesign[i]=='+'){
					if((q+e)>=100||(q+e)<=0){
						flag=1;
						continue;
					}
					else{
						anwser[i]=q+e;
						qq[i]=q;
						ee[i]=e;
					}
	
				}
				else if(usesign[i]=='-'){
					if((q-e) >= 100||(q-e) <= 0){
						flag = 1;
						continue;
					}
					else{
						anwser[i]=q-e;
						qq[i]=q;
						ee[i]=e;
					}
				}
			}while(flag==1); 
		}
	for(i=0;i<10;i++)
	{
		printf("%d %c %d=",qq[i],usesign[i],ee[i]);
		scanf("%d",&myan[i]);
		if(myan[i]==anwser[i])
		{
			point+=1;
		}
	}
	printf("����     |��ȷ��| ��Ĵ�\n");
	for(i=0;i<10;i++)
	{
		printf("%d%c%d\t |   %d\t  | %d\t\n",qq[i],usesign[i],ee[i],anwser[i],myan[i]);
	}
	return point;
}


