#include<iostream>

template<bool test, class T = void>
struct enableif{};

template<class T>
struct enableif<true, T>{
    using type = T;
};



int main(){
    // auto x = static_cast<enableif<2==3, int>::type*>3;
}