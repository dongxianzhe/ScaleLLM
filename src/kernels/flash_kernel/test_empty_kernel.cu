#include<iostream>

__global__ void empty_kernel(){

}

int main(){
    empty_kernel<<<1, 32>>>();
}