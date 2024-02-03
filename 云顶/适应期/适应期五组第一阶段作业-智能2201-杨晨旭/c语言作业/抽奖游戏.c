#include<stdio.h>
#include<stdlib.h>
#include<time.h>
int main() {
	srand((unsigned int)time(NULL));
	int special = rand() %10;
	int specialNum = rand()% 4 +1 ;
	int count;
	int winNumber;
	printf("特征数为%d,特征数个数为%d个\n",special,specialNum);
	int i;
	for(i =1000; i<=9999; i++) {
		count = 0; 
		int temp = i;
		while(1) {
			int c = temp % 10;
			if( c == special) {
				count++;
			}
			if(temp<10) {
				break;
			}
			temp = temp / 10;
		}
//		i1=i%10;
//		i2=(i/10)%10;
//		i3=(i/100)%10;
//		i4=i/1000;
//		if (i1==n){
//			N1+=1;
//		}
//		if (i2==n){
//			N1+=1;
//		}
//		if (i3==n){
//			N1+=1;
//		}
//		if (i4==n){
//			N1+=1;
//		}
		if(count==specialNum) {
			printf("%d ",i);
			winNumber++;
		}
	}
	printf("\n");
	printf("一共有%d个",winNumber);
	return 0;
	getchar();
	getchar();
	getchar();
	getchar();
}
