#include<iostream>
#include<torch/torch.h>

void add(torch::Tensor a, torch::Tensor b, torch::Tensor& c){
    c = a + b;
}

int main(){
    auto a = torch::ones({10}, torch::dtype(torch::kFloat));
    auto b = torch::ones({10}, torch::dtype(torch::kFloat));
    auto c = torch::zeros({10}, torch::dtype(torch::kFloat));
    add(a, b, c);
    puts("------------ a ------------------");
    std::cout << a << std::endl;
    puts("------------ b ------------------");
    std::cout << b << std::endl;
    puts("------------ c ------------------");
    std::cout << c << std::endl;
    return 0;
}