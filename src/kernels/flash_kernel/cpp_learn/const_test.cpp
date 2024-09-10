#include<iostream>
#include<string>

const int v1 = 1;
int v2 = 2;
std::string s1 = "abc";
const std::string s2 = "def";

class A{
public:
    const int* p1 = &v1;
    int* p2 = &v2;
    const int& r1 = v2;
    int& r2 = v2;
    int m1 = 6;
    const int m2 = 5;

    void fun() const{
        r2 ++;
    }
};

int main(){
    const A o1;
    A o2;

    o1.fun();

    printf("%d %d %d %d\n", *o1.p1, *o1.p2, o1.m1, o1.m2);

    s1[2] = 'z';

    std::cout << s1 << std::endl;
    std::cout << s2 << std::endl;
    
    return 0;
}