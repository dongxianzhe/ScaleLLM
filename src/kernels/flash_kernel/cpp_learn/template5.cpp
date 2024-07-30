#include<iostream>
using namespace std;

template<typename... Args>
struct sum;

template<typename first, typename... rest>
struct sum<first, rest...>{
    static constexpr first value = sum<first>::value + sum<rest...>::value;
};

template<typename first, typename last>
struct sum<first, last>{
    static constexpr last value = sizeof(first) + sizeof(last);
};

template<typename last> // 这个特化不会用到，因为在两个参数的时候就会匹配到上一个模板
struct sum<last>{
    static constexpr last value = sizeof(last);
};

template<>
struct sum<>{
    static constexpr int value = 0;
};

int main(){
    printf("%d\n", sum<int, char, float>::value);
    printf("%d\n", sum<>::value);
    return 0;
}
