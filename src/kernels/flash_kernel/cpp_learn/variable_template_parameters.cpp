#include <iostream>
#include <tuple>

// Variadic template class
template <class... T>
class MyClass {
public:
    // Constructor that takes a variable number of arguments
    MyClass(T... args) : data(args...) {}

    // Function to print the arguments
    void print() {
        printTuple(data);
    }

private:
    std::tuple<T...> data;

    // Helper function to print the tuple elements
    template <std::size_t... I>
    void printHelper(std::index_sequence<I...>) {
        ((std::cout << std::get<I>(data) << " "), ...);
        std::cout << std::endl;
    }

    void printTuple(const std::tuple<T...>& t) {
        printHelper(std::make_index_sequence<sizeof...(T)>());
    }
};

int main() {
    MyClass<int, double, std::string> myObj(1, 2.5, "Hello");
    myObj.print(); // Output: 1 2.5 Hello
    return 0;
}