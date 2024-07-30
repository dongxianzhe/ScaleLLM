#include<iostream>

template<int... Ints>
using int_sequence = std::integer_sequence<int, Ints...>;

template<size_t ... Ints>
using index_sequence = std::index_sequence<Ints...>;

template<size_t N, class T>
struct EBO{};

template<class IndexSeq, class... T>
struct TupleBase;

template<size_t... I, class... T>
struct TupleBase<index_sequence<I...>, T...> : EBO<I, T>...{
    template<class... U>
    constexpr explicit TupleBase(U const&... u) : EBO<I, T>(u)...{}
};


int main(){
    // int res = sum(std::pair<int, int>(2, 3));
    // printf("%d\n", res);
    // printf("%d\n", sum(2));
    return 0;
}