#include<stdio.h>
#include<string.h>
int search(char arr1[],char arr2[],int,int) ;
int main()
{

	char parent[1024];
	char son[1024];
	printf("������һ���ַ���\n");
	scanf("%s",&parent); 
	printf("������һ����Ҫ���ҵ������Ӵ�\n");
	scanf("%s",&son); 
	int l1 = strlen(parent);
	int l2 = strlen(son);
	char*p1=parent;
	char*p2=son;
	printf("�Ӵ��ڸ��ַ�����һ��������%d��",search(parent,son,l1,l2)); 
	return 0;
}
int search(char*p1,char*p2,int l1,int l2){ 
	int num = 0;
	int n=0;
	int i;
	for (i = 0;i < l1;i++){  
		int index = i;
		int n=0;
		int l;
		for ( l = 0;l < l2;l++){ 
			if(*(p1+index++) == *(p2+l)){
				n++;
				continue;
			}			
			else{
				break;
			}
		}
		if(n == l2){
			num++;
		}

	}
	return num;
}
