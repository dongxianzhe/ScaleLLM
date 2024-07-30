#include<iostream>
#include<array>


template<int... I>
struct seq{
    using type = seq;
};

template<class... I>
struct concat;

template<int L, int... H>
struct concat<seq<L>, seq<H...> > : public seq<L, (H + 1) ...> {};

template<int N>
struct make : public concat<seq<0>, typename make<N-1>::type> {
};

template<>
struct make<1> : public seq<0> {
};

template<>
struct make<0> : public seq<> {
};

template<int N, int F, int...I>
constexpr int get(seq<F, I...> q) {
    if constexpr (N == 0) {
        return F;
    } else {
        return get<N - 1>(seq<I...>{});
    }
}

void make_test() {
    auto seq = make<4>{};
    static_assert(get<0>(seq) == 0, "0");
    static_assert(get<1>(seq) == 1, "1");
    static_assert(get<2>(seq) == 2, "2");
    static_assert(get<3>(seq) == 3, "3");
}

int main(){
    make_test();
    // concat<seq<0>, seq<4, 5>>::type;
    return 0;
}