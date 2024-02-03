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
		printf("选择游戏模式：1.自动生成特征数 2.手动输入特征数\n");
		printf("请输入1或2:\n");
		scanf("%d",&choice);
		while (i==1) {
			if (choice==1) {
				a=rand()%5+5;
				printf("你抽取的数字是%d\n",a);
				break;
			} else if(choice==2) {
				while (i==1) {
					printf("请输入特征数(5~9):");
					scanf("%d",&a);
					if(a<5||a>9) {
						printf("你输入的不符合要求\n");
					} else {
						break;
					}
				}
				break;
			} else {
				continue;
				printf("输入不符合规则，请重新输入");
			}
		}
		printf("1-100中过它的数有\n");
		while(n<100) {
			n+=1;
			if(n%a != 0&&n%10!=a&&n/10!=a) {
				printf("%d ",n);
				num+=1;
			}

		}
		printf("特征数为%d",a);
		printf("共有%d个数字!",num);
		printf("\n");
		n=0;
	}
	return 0;
}
