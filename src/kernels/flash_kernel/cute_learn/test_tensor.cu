#include<iostream>
#include<torch/torch.h>
#include<cute/layout.hpp>
#include<cute/stride.hpp>
#include<cute/tensor.hpp>
#include<thrust/device_vector.h>

using namespace cute;

template<typename T>
__global__ void kernel(T* ptr){
    // ((_3,2),(2,_5,_2)):((4,1),(_2,13,100))
    Tensor dA = make_tensor(make_gmem_ptr(ptr), make_shape (make_shape (Int<3>{},2), make_shape (       2,Int<5>{},Int<2>{})),
                                make_stride(make_stride(       4,1), make_stride(Int<2>{},      13,     100)));
    // ((2,_5,_2)):((_2,13,100))
    Tensor dB = dA(2,_);
    // ((_3,_2)):((4,1))
    Tensor dC = dA(_,5);
    // (_3,2):(4,1)
    Tensor dD = dA(make_coord(_,_),5);
    // (_3,_5):(4,13)
    Tensor dE = dA(make_coord(_,1),make_coord(0,_,1));
    // (2,2,_2):(1,_2,100)
    Tensor dF = dA(make_coord(2,_),make_coord(_,3,_));
    printf("dA: ");print(dA); printf("\n");
    printf("dB: ");print(dB); printf("\n");
    printf("dC: ");print(dC); printf("\n");
    printf("dD: ");print(dD); printf("\n");
    printf("dE: ");print(dE); printf("\n");
    printf("dF: ");print(dF); printf("\n");
}

int main(){
    puts("---------- tensor access --------------------");
    {
        float* ptr;
        // ((_3,2),(2,_5,_2)):((4,1),(_2,13,100))
        Tensor A = make_tensor(ptr, make_shape (make_shape (Int<3>{},2), make_shape (       2,Int<5>{},Int<2>{})),
                                    make_stride(make_stride(       4,1), make_stride(Int<2>{},      13,     100)));

        // ((2,_5,_2)):((_2,13,100))
        Tensor B = A(2,_);

        // ((_3,_2)):((4,1))
        Tensor C = A(_,5);

        // (_3,2):(4,1)
        Tensor D = A(make_coord(_,_),5);

        // (_3,_5):(4,13)
        Tensor E = A(make_coord(_,1),make_coord(0,_,1));

        // (2,2,_2):(1,_2,100)
        Tensor F = A(make_coord(2,_),make_coord(_,3,_));
        printf("A: ");print(A); printf("%d \n", A.data() - ptr);
        printf("B: ");print(B); printf("%d \n", B.data() - ptr);
        printf("C: ");print(C); printf("%d \n", C.data() - ptr);
        printf("D: ");print(D); printf("%d \n", D.data() - ptr);
        printf("E: ");print(E); printf("%d \n", E.data() - ptr);
        printf("F: ");print(F); printf("%d \n", F.data() - ptr);
    }

    puts("--------- device tensor ---------------------");

    // thrust::device_vector<float> dv(120);
    // kernel<<<1, 1>>>(thrust::raw_pointer_cast(dv.data()));
    
    puts("---------- zipped divided --------------------");
    {
        torch::Tensor t = torch::arange(8 * 24, torch::kInt32);
        int* ptr = static_cast<int*>(t.data_ptr());
        Tensor A = make_tensor(ptr, make_shape(8,24));  // (8,24)
        auto tiler = Shape<_4,_8>{};                    // (_4,_8)

        Tensor tiled_a = zipped_divide(A, tiler);       // ((_4,_8),(2,3))
        
        printf("A: ");print_tensor(A);
        printf("tiler: ");print(tiler);puts("");
        printf("tiled_a: ");print_tensor(tiled_a);
        for(int i = 0;i < 2;i ++){
            for(int j = 0;j < 3; j++){
                printf("local_tile (%d, %d)", i, j);print_tensor(local_tile(A, tiler, make_coord(i, j)));
            }
        }
        for(int i = 0;i < 2;i ++){
            printf("local_tile (%d, _)", i);print_tensor(local_tile(A, tiler, make_coord(i, _)));
        }
        for(int j = 0;j < 3;j ++){
            printf("local_tile (_, %d)", j);print_tensor(local_tile(A, tiler, make_coord(_, j)));
        }
        for(int i = 0;i < 8;i ++){
            printf("outer_parition %d", i);print_tensor(outer_partition(A, tiler, i));
        }
        for(int i = 0;i < 8;i ++){
            printf("local_partition %d", i);print_tensor(local_partition(A, make_layout(make_shape(4, 8), make_stride(1, 4)), i));
        }
        for(int i = 0;i < 8;i ++){
            printf("local_partition %d", i);print_tensor(local_partition(A, make_layout(make_shape(4, 8), make_stride(8, 1)), i));
        }
        // printf("local_partition larger: ");print_tensor(local_partition(A, make_layout(make_shape(4, 64), make_stride(64, 1)), 0));
    }
    puts("------------------------------");
    return 0;
}