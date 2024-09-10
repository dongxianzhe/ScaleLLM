#include<iostream>

int x = 2;

int& getValue(){
    return x;
}

class A{
public:
    int m1 = 2;
    const int& operator[](int index) const{
        printf("operator const");
        return m1;
    }

    // int& operator[](int index) {
    //     printf("operator no const");
    //     return m1;
    // }
};

int main(){

    printf("%d\n", x);
    getValue() = 3;
    printf("%d\n", x);

    A o1;
    printf("%d\n", o1.m1);
    printf("%d\n", o1[2]);
    // o1[3] = 4;
    printf("%d\n", o1.m1);
    return 0;
}