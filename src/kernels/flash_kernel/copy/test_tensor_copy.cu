#include<torch/torch.h>

int main(){
    auto a = torch::randn({3, 3}, torch::device(torch::kCUDA));
    auto b = a;
    std::cout << "b before: " <<  b << std::endl;
    a[0][0] = 4;
    std::cout << "b after: " <<  b << std::endl;
}