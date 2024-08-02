#include<iostream>

template<class T>
T x;



int main(){
    x<int> = 2;
    x<int> = 4;

    x<float> = 2.3;
    printf("%d\n", x<int>);
    printf("%f\n", x<float>);
}