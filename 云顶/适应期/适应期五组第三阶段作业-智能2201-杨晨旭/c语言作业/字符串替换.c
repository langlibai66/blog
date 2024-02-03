#include<stdio.h>
#include<string.h>
int move(char[],int); //×Ö·û´®Ãû ²åÈë×Ö·û ²åÈëÎ»ÖÃ 
char *plus;
int main()
{
	char str[1000]={};
	printf("please input a string:\n");
	gets(str) ;
	int n=strlen(str);
	for (int i=0;i<n;i++)
	{
		if(str[i]==' '){
			int index=i;
			move(str,index);
			move(str,index);
			str[index]='%';
			str[index+1]='4';
			str[index+2]='0';
			n+=2;
		}
	}
	printf("%s",str);
	return 0;
}
int move(char arr[],int l){//×Ö·û´®Ãû ²åÈë×Ö·û ²åÈëÎ»ÖÃ 
	int n=strlen(arr);
	for(int i = n;i > l;i--){
			arr[i]=arr[i-1];
		}
	return 0;
	}



