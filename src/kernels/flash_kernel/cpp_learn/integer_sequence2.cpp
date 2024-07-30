#include<iostream>
#include<array>

template<size_t... N>
static constexpr int square_nums(int index, std::index_sequence<N...>){
    constexpr int nums[] = {N * N ...};
    return nums[index];
}

template<int N>
constexpr static int const_nums(int index){
    return  square_nums(index, std::make_index_sequence<N>{});
}

int main(){
    static_assert(const_nums<10>(2) == 4, "err 1");
}