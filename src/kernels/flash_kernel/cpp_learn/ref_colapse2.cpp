#include <type_traits>
#include <iostream>

using namespace std;

template <typename T>
void Func2(T&& j) {
    cout << is_rvalue_reference<T&&>::value << endl;
}

template <typename T>
void Func1(T&& i) {
    cout << is_rvalue_reference<T&&>::value << endl;
    Func2(i);
}

int main() {
    Func1(10);
    return 0;
}