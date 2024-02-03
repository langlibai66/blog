#include<stdio.h>
#include<string.h>
fun(int x)
{
	int p;
	if(x==1)
	return(3);
	p=x-fun(x-2);
	return p;
}
int main()
{
	printf("%d",fun(9));
	return 0;
}
