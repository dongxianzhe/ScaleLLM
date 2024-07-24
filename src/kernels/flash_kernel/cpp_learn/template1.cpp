#include<iostream>

template<class T>
T maxx(T a, T b){
    if(a < b)return b;
    else return a;
}

struct A{
    int value; 
    bool operator<(A o){ // 没有这个函数编译会报错
        return value < o.value; 
    }
};

int main(){
    int x = maxx(2, 3);
    printf("%d \n", x);

    A a, b;
    A c = maxx(a, b);
    return 0;
}