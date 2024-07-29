#include<iostream>

void print(){

}

template<typename T, typename... Types>
void print(T firstArg, Types... args){
    std::cout << firstArg << std::endl;
    print(args...);
}


int main(){
    std::string s = "world";
    print(2, "hello", s);
    return 0;
}