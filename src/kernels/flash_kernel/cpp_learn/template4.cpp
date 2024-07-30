#include<iostream>

template<class... Types>
struct tuple{
    tuple(Types... t){
        
    }
    void print(){}

    template<class PT, class... PTypes>
    void print(PT a, PTypes... b){
        std::cout << a << ", ";
        print(b...);
    }
};

template<class... Shapes>
using Shape = tuple<Shapes...>;

// void print(){

// }

// template<class T, class... Types>
// void print(T head,Types... rest){
//     std::cout << head << std::endl;
//     print(rest...);
// }

template<class T>
void show(T a){
    std::cout << a << std::endl;
}

template<class... T>
void print(T... v){
    // int arr[] = {(show(v), 0)...};
}

template<class T>
T sum(T t){
    return t;
}
template<class T, class... Types>
T sum(T first ,Types... rest){
    return first + sum<T>(rest...);
}

int main(){
    // auto t = tuple<int, std::string, std::string>(2, "hello", "world");
    // t.print();
    print("hello", 2 , "world");puts("");
    // printf("%d\n", sum("hello", 2, 3));
    printf("%d\n", sum(1, 2, 3));
    return 0;
}