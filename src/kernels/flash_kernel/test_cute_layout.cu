#include<include/cute>

int main(){
    Layout s2xd4_row = make_layout(make_shape(Int<2>{}, 4), LayoutRight{});  
    cute::print(s2xd4_row);
    return 0;
}