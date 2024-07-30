#include<iostream>
struct A{
    int b, a;
    // A(int a, int b){
    //     a = a;b = b;
    // }
};

int main(){
    A x {2, 3} ;
    printf("%d %d\n", x.a, x.b);

    return 0;
}