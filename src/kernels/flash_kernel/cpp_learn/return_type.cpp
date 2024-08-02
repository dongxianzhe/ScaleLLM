#include<iostream>

int arr[3] = {1, 2, 3};
int (*a)[3] = &arr;

int (*fun())[3]{
    return a;
}

auto fun2() -> int(*)[3]{
    return a;
}

// int(*)[3] fun(){ error
//     return a;
// }

int main(){
    auto p = static_cast<int (*(*)())[3]>(nullptr);

}