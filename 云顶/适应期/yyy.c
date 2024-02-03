#include<stdio.h>
int main()
{
int arr[7][7]={1};
int i,j;
for(i=0;i<7;i++)
{
	arr[i][0]=1;
	arr[i][i]=1;
}
for(i=2;i<7;i++)
{
	for(j=1;j<i;j++){
	arr[i][j]=arr[i-1][j-1]+arr[i-1][j];
	}
}
for(i=0;i<7;i++)
{
	for(j=0;j<=i;j++)
	printf("%4d",arr[i][j]);
	printf("\n");
}

