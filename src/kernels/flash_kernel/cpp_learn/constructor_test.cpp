#include<iostream>

class A{
public:
    A(){
        static int cnt = 0;
        cnt ++;
        printf("A default construct %d \n", cnt);
    }
    A(const A& o){
        printf("A copy constructor\n");
    }
    A& operator=(const A& o){
        printf("A operator=\n");
        return *this;
    }
    A(const A&& o){
        printf("A move constructor\n");
    }
    A& operator=(const A&& o){
        printf("A move operator=\n");
        return *this;
    }
};

class B{
public:
    const int m1 = 2;
    A o1;
};

A fun(){
    A a;
    return a;
}

int main(){
    A o1, o2;
    // o1 = std::move(o2);
    o1 = fun();
    o1 = A();
    o1 = o2;
}