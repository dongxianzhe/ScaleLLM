#include<iostream>
#include<torch/torch.h>
#include<cute/layout.hpp>
#include<cute/stride.hpp>
#include<cute/tensor.hpp>

int main(){
    float a[10][10];
    for(int i = 0;i < 10;i ++){
        for(int j = 0;j < 10;j ++){
            a[i][j] = i * 10 + j;
        }
    }
    using namespace cute;
    Tensor A = make_tensor((float*)a, make_shape(10, 10), make_stride(10, 1));
    for(int i = 0;i < 10;i ++){
        for(int j = 0;j < 10;j ++){
            printf("%f ", A(i, j));
        }puts("");
    }

    return 0;
}