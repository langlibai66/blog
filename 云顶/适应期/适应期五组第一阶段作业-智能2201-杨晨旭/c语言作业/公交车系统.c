#include<stdio.h>
#include<stdlib.h>
#include<time.h>

int main()
{
	srand((unsigned int)time(NULL));
	int hour1,min1,hour2,min2,p1,p2,m1,m2;
	hour1=rand()%3;
	min1=rand()%59;
	p1=0;
	printf("������ʼ����%d.%d\n",hour1+9,min1);
	printf("������С����ʼ��ʱ����վ̨\n");
	printf("��������10 40 2\n");		
	while (p1==0){
		printf("�û�����");
		scanf("%d %d %d",&hour2,&min2,&p2);
		if(hour2>=9&&hour2<=12&&min2>=0&&min2<=59&&p2>0&&p2<=2){
			break;			 
		}
		else{
			printf("������Ĳ����ϸ�ʽ\n����������\n");
			continue;
		} 
	}
	m1=hour1*60+min1;
	m2=(hour2-9)*60+min2;
	int dp =p2-p1;
	int i=(m2-m1-dp*20);
	int n =(i/70)*70-i;
	if(m2-m1-20*dp<(-70)){
		n+=70;
	}
	if((m2-m1)>20*dp)
	{
		n+=70;
	}


	printf("�ȴ���ʱ����%dmin",n);
	
	return 0;
}
