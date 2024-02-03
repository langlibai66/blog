#include<stdio.h>
#define Length 10
int a[Length];
void input()
{
	printf("请输入10个整数：\n");
	for(int i=0;i<Length;i++)
	{
		scanf("%d",&a[i]);
	}
}
void output()
{
	printf("请输出数组：\n");
	for(int i=0;i<Length;i++)
	{
		printf("%d ",a[i]);
	}
	printf("\n");
}
void swap(int i,int j)
{
	int tmp;
	tmp=a[i];
	a[i]=a[j];
	a[j]=tmp;
}
void maxmin()
{
	int max=0,min=0;
	for(int i=1;i<Length;i++)
	{
		if(a[i]<a[min])
		{
			min=i;
		}
	}
	if(min!=0)
	{
		swap(min,0);
	}
	for(int j=1;j<Length;j++)
	{
		if(a[j]>a[max])
		{
			max=j;
		}
	}
	if(max!=Length-1)
	{
		swap(max,Length-1);
	}
}
int main()
{
	input();
	maxmin();
	output();
	return 0;
}
