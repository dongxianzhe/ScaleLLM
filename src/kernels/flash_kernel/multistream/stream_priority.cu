#include<iostream>

int main(){
    int priority_low, priority_high;

    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);

    printf("priority_low, priority_high = %d %d\n", priority_low, priority_high);

    return 0;
}