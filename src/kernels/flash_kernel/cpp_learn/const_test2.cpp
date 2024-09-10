#include<iostream>

class A{
public:
    int m[3] = {0, 1, 2};
    int m2 = 4;
    int& m3 = m2;
};


int main(){
    const A o1;
    o1.m3 = 6;
    printf("%d\n", o1.m3);
    for(int i = 0;i < 3;i ++)printf("%d ", o1.m[i]);puts("");

}
