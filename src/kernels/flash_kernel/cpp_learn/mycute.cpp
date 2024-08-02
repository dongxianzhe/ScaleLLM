#include<iostream>
// :const
template<auto v>
struct C{
    using type = C<v>;
    static constexpr auto value = v;
    using value_type = decltype(v);
    constexpr operator value_type() const noexcept{return value;}
    constexpr value_type operator()() const noexcept {return value;}
};

template<bool b>
using bool_constant = C<b>;
using true_type = bool_constant<true>;
using false_type =bool_constant<false>;

template<class T, T v>
struct integral_constant : C<v>{
    using type = integral_constant<T,v>;
    static constexpr T value = v;
    using value_type = T;
    constexpr value_type operator()(){return value;}
};

// :tuple
template<size_t N, class T, bool IsEmpty = std::is_empty<T>::value>
struct EBO;

template<size_t N, class T>
struct EBO<N, T, true>{
    using type1 = int;
    constexpr EBO(T const&){}
};

template<size_t N, class T>
struct EBO<N, T, false>{
    using type2 = float;
    T t_;
    template<class U>
    constexpr EBO (U const& u) : t_{u} {}
};

template<class IndexSeq, class... T>
struct TupleBase;

template <size_t... I, class... T>
struct TupleBase<std::index_sequence<I...>, T...> : EBO<I,T>...{
    template<class... U>
    constexpr explicit TupleBase(U const&... u) : EBO<I,T>(u)... {} 
};

template<class... T>
struct tuple : TupleBase<std::make_index_sequence<sizeof...(T)>, T...> {
    template <class... U>
    tuple(U const&... u) : TupleBase<std::make_index_sequence<sizeof...(T)>, T...>(u...) {}};


// :tuple algorithm
template<class... T>
tuple<T...> make_tuple(T const& ... t){return {t...};}

template <class T, class = void>
struct tuple_size;

template<class T, T... Ints>
struct tuple_size<std::integer_sequence<T, Ints...>> : integral_constant<size_t, sizeof...(Ints)>{};

// :layout shape stride coord step
template<class... Shapes>
using Shape = tuple<Shapes...>;

template<class... Ts>
constexpr Shape<Ts...> make_shape(Ts const&... t){return {t...};}

template<class... Strides>
using Stride = tuple<Strides...>;

template<class... Ts>
constexpr Stride<Ts...> make_stride(Ts const&... t){return {t...};}

template<class... Layouts>
using Tile = tuple<Layouts...>;

template<class... Ts>
constexpr Tile<Ts...> make_tile(Ts const&... t){return {t...};}

template<class Shape, class Stride>
struct Layout : private tuple<Shape, Stride>{
    constexpr Layout(Shape const& shape = {}, Stride const& stride = {}) : tuple<Shape, Stride>(shape, stride) {}

    // template<class Coord>
    // constexpr auto operator()(Coord const& coord) const{
    //     return crd2idx(coord, shape(), stride())
    // }
};



int main(){
    auto x = tuple<int, int, int, int>(2, 3, 4, 4);
    EBO<0, C<0>>(C<0>{});

    printf("%d\n", tuple_size<std::integer_sequence<uint32_t, 1, 2, 2, 2, 2>>::value);

    return 0;
}