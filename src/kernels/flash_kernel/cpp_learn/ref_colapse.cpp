#include<iostream>

template<class T>
void print(T&& t){
    printf("%d\n", t);
}

template<>
void print<int>(int&& t){
    printf("int %d\n", t);
}

template<>
void print<int&>(int& t){
    printf("int& %d\n", t);
}

void fun0(int a){
    puts("fun0");
}

void fun1(int& a){
    puts("fun1");
}

void fun2(int&& a){
    puts("fun2");
}


int main(){
    int a = 10;
    int& b = a;
    int&& c = 11;
    int& d = c;

    print(a);
    print(b);
    print(c);
    print(12);
    print(static_cast<int&&>(12));

    fun1(c);

    return 0;
}