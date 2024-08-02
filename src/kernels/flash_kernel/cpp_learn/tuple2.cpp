#include<iostream>
#include<tuple>

int main(){
    {
        puts("------------ tuple create and access ------------------");
        std::string s1 = "hello";
        std::string s2 = "world";
        std::tuple<std::string, std::string> t1(s1, s2);

        std::cout << std::get<0>(t1) << std::endl;
        std::cout << std::get<1>(t1) << std::endl;
    }
    {
        puts("------------tuple size------------------");
        std::tuple<char, int, long, std::string> t1('A', 2, 3, "4");
        std::cout << std::tuple_size<decltype(t1)>::value << std::endl;
    }
    {
        puts("---------get element type---------------------");
        std::tuple<int, std::string> t(9, std::string("abc"));
        std::tuple_element<0, decltype(t)>::type i = std::get<0>(t);
        std::cout << i << std::endl;
    }
    return 0;
}