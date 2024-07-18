#include<cute/layout.hpp>
#include <cute/tensor.hpp>

using namespace cute;

int main()
{
  using MMA = MMA_Traits<SM80_16x8x8_F32F16F16F32_TN>;
  print("ALayout: "); print(typename MMA::ALayout{}); print("\n");
  print("BLayout: "); print(typename MMA::BLayout{}); print("\n");
  print("CLayout: "); print(typename MMA::CLayout{}); print("\n");
}