#include<iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

int main(){
    thrust::host_vector<float> h_A(10);
    for (int i = 0; i < 10; i ++) h_A[i] = i;
    for(int i = 0;i < 10;i ++)printf("%f ", h_A[i]);puts("");
    return 0;
}