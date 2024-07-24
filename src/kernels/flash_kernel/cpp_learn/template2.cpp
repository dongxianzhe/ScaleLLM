#include<iostream>

template<typename T>
T fun(){
    return 2;
}

int main(){
    int x = fun<int>();
    printf("%d \n",x);
    return 0;
}