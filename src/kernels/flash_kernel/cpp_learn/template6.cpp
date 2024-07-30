#include<iostream>

using namespace std;

template<int...>
struct IndexSeq{};

template<int N, int... Indexes>
struct make_indexes : make_indexes<N - 1, N - 1, Indexes...>{};

template<int... Indexes>
struct make_indexes<0, Indexes...>{
    using type = IndexSeq<Indexes...>;
};

int main(){
    using T = make_indexes<3>::type;
    std::cout << typeid(T).name() << endl;
}