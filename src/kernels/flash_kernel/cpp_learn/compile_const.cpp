#include<iostream>

template<auto v>
struct C{
    using type = C<v>;
    static constexpr auto value = v;
    using value_type = decltype(v);
    inline constexpr operator value_type() const noexcept { puts("convert called");return value;}
    inline constexpr value_type operator()() const noexcept {puts("() called");return value;}
    inline constexpr value_type operator()(int x) const noexcept {return x;}
    int size(){return sizeof(value);}
};

template<class T, T v>
struct integral_constant : C<v>{
    using type = integral_constant<T,v>;
    static constexpr T value = v;
    using value_type = T;
    inline constexpr value_type operator()() const noexcept {
        return value;
    }
};

template <bool b>
using bool_constant = C<b>;

using true_type  = bool_constant<true>;
using false_type = bool_constant<false>;

template <auto n, class T>
struct is_constant : false_type {};
// template <auto n, class T>
// struct is_constant<n, T const > : is_constant<n,T> {};
// template <auto n, class T>
// struct is_constant<n, T const&> : is_constant<n,T> {};
// template <auto n, class T>
// struct is_constant<n, T      &> : is_constant<n,T> {};
// template <auto n, class T>
// struct is_constant<n, T     &&> : is_constant<n,T> {}; // 这是个右值引用
// template <auto n, auto v>
// struct is_constant<n, C<v>                  > : bool_constant<v == n> {};
template <auto n, class T, T v>
struct is_constant<n, integral_constant<T,v>> : bool_constant<v == n> {};

int main(){
    {
        puts("------------is constant------------------");
        int x = 2;
        printf("is constant %d\n", is_constant<2, int>::value);

    }
    {
        puts("---------- compiler constant --------------------");
        C<2> c;
        int x = c;
        printf("%d\n", x);
        printf("%d\n", c);
        printf("%d\n", c());

    }
    {
        puts("---------- size--------------------");
        C<true> a;
        C<3> b;
        C<'a'> c;
        constexpr short y = 0;
        C<y> d;
        printf("a size = %d \n",a.size());
        printf("b size = %d \n",b.size());
        printf("c size = %d \n",c.size());
        printf("d size = %d \n",d.size());

    }
    {
        puts("------------test operator ()------------------");
        C<2> a;
        for(int i = 0;i < 10;i ++)printf("%d ",a(i));
    }
    {
        int x = 2;
        int& y = x;
        int&& z = 2;
        y = z;
    }
    return 0;
}