#include<stdio.h>
#include<stdlib.h>
#include<time.h>

int main() {
	srand((unsigned int)time(NULL));
	int a,n,num;
	int i=1;
	int choice;
	int l;
	while (l==0) {
		printf("ѡ����Ϸģʽ��1.�Զ����������� 2.�ֶ�����������\n");
		printf("������1��2:\n");
		scanf("%d",&choice);
		while (i==1) {
			if (choice==1) {
				a=rand()%5+5;
				printf("���ȡ��������%d\n",a);
				break;
			} else if(choice==2) {
				while (i==1) {
					printf("������������(5~9):");
					scanf("%d",&a);
					if(a<5||a>9) {
						printf("������Ĳ�����Ҫ��\n");
					} else {
						break;
					}
				}
				break;
			} else {
				continue;
				printf("���벻���Ϲ�������������");
			}
		}
		printf("1-100�й���������\n");
		while(n<100) {
			n+=1;
			if(n%a != 0&&n%10!=a&&n/10!=a) {
				printf("%d ",n);
				num+=1;
			}

		}
		printf("������Ϊ%d",a);
		printf("����%d������!",num);
		printf("\n");
		n=0;
	}
	return 0;
}
