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
	printf("������15��ѧ����ѧ�ţ�������3�ſεĳɼ����Կո�ָ����Իس�����һ�е��������룩��\n");
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
	printf("��߷�ѧ����������->ѧ��:%s\n����:%s\n3�ſγ̳ɼ�:%d %d %d\n", grade[ret].a, grade[ret].b, grade[ret].c, grade[ret].d, grade[ret].e);
	return 0;

}
