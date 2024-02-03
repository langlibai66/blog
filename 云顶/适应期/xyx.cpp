#include<stdio.h>
#include<string.h>
int main()
{
	char a[5][80];
	char(*p)[80];
	char* temp = a[0];
	int i;
	p = a;
	for (i = 0; i < 5; i++)
	{
		scanf("%s", (p + i));
	}
    for (i = 0; i < 5; i++)
    {
        if (strcmp(temp, *(p + i)) <0)
        {
            strcpy(temp, *(p + i));
        }
    }
    printf("较大的字符串是%s\n", temp);
    return 0;
}
