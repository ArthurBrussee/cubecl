typedef unsigned int uint;

struct __align__(16) float_4 {
  float i_0;
  float i_1;
  float i_2;
  float i_3;
};

extern "C" __global__ void kernel(float_4 input_0[], float_4 input_1[],
                                  float_4 output_0[], uint info[]) {

  int3 absoluteIdx = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                               blockIdx.y * blockDim.y + threadIdx.y,
                               blockIdx.z * blockDim.z + threadIdx.z);

  uint idxGlobal =
      (absoluteIdx.z * gridDim.x * blockDim.x * gridDim.y * blockDim.y) +
      (absoluteIdx.y * gridDim.x * blockDim.x) + absoluteIdx.x;
  uint rank = info[0];
  uint rank_2 = rank * 2;
  uint l_0_0;
  bool l_0_1;
  bool l_0_2;
  float_4 l_0_3;
  float_4 l_0_4;
  l_0_0 = info[(3 * 2 * info[0]) + 3] / 4;
  l_0_1 = idxGlobal < l_0_0;
  if (l_0_1) {

    for (uint l_2_0 = uint(0); l_2_0 < uint(256); ++l_2_0) {
      l_0_0 = l_2_0 % uint(2);
      l_0_2 = l_0_0 == uint(0);
      if (l_0_2) {
        uint l_3_0;
        bool l_3_1;
        l_3_0 = info[(3 * 2 * info[0]) + 1] / 4;
        l_3_1 = idxGlobal < l_3_0;
        if (l_3_1) {
          l_0_3 = input_0[idxGlobal];
        } else {
          l_0_3.i_0 = float(0.0);
          l_0_3.i_1 = float(0.0);
          l_0_3.i_2 = float(0.0);
          l_0_3.i_3 = float(0.0);
        }
        uint l_3_2;
        bool l_3_3;
        l_3_2 = info[(3 * 2 * info[0]) + 2] / 4;
        l_3_3 = idxGlobal < l_3_2;
        if (l_3_3) {
          l_0_4 = input_1[idxGlobal];
        } else {
          l_0_4.i_0 = float(0.0);
          l_0_4.i_1 = float(0.0);
          l_0_4.i_2 = float(0.0);
          l_0_4.i_3 = float(0.0);
        }
        l_0_3.i_0 = l_0_3.i_0 * l_0_4.i_0;
        l_0_3.i_1 = l_0_3.i_1 * l_0_4.i_1;
        l_0_3.i_2 = l_0_3.i_2 * l_0_4.i_2;
        l_0_3.i_3 = l_0_3.i_3 * l_0_4.i_3;
        l_0_3.i_0 = cos(l_0_3.i_0);
        l_0_3.i_1 = cos(l_0_3.i_1);
        l_0_3.i_2 = cos(l_0_3.i_2);
        l_0_3.i_3 = cos(l_0_3.i_3);
        uint l_3_4;
        bool l_3_5;
        l_3_4 = info[(3 * 2 * info[0]) + 3] / 4;
        l_3_5 = idxGlobal < l_3_4;
        if (l_3_5) {
          l_0_4 = output_0[idxGlobal];
        } else {
          l_0_4.i_0 = float(0.0);
          l_0_4.i_1 = float(0.0);
          l_0_4.i_2 = float(0.0);
          l_0_4.i_3 = float(0.0);
        }
        l_0_4.i_0 = l_0_4.i_0 - l_0_3.i_0;
        l_0_4.i_1 = l_0_4.i_1 - l_0_3.i_1;
        l_0_4.i_2 = l_0_4.i_2 - l_0_3.i_2;
        l_0_4.i_3 = l_0_4.i_3 - l_0_3.i_3;
        uint l_3_6;
        bool l_3_7;
        l_3_6 = info[(3 * 2 * info[0]) + 3] / 4;
        l_3_7 = idxGlobal < l_3_6;
        if (l_3_7) {
          output_0[idxGlobal] = l_0_4;
        }
      } else {
        uint l_3_0;
        bool l_3_1;
        l_3_0 = info[(3 * 2 * info[0]) + 1] / 4;
        l_3_1 = idxGlobal < l_3_0;
        if (l_3_1) {
          l_0_4 = input_0[idxGlobal];
        } else {
          l_0_4.i_0 = float(0.0);
          l_0_4.i_1 = float(0.0);
          l_0_4.i_2 = float(0.0);
          l_0_4.i_3 = float(0.0);
        }
        uint l_3_2;
        bool l_3_3;
        l_3_2 = info[(3 * 2 * info[0]) + 2] / 4;
        l_3_3 = idxGlobal < l_3_2;
        if (l_3_3) {
          l_0_3 = input_1[idxGlobal];
        } else {
          l_0_3.i_0 = float(0.0);
          l_0_3.i_1 = float(0.0);
          l_0_3.i_2 = float(0.0);
          l_0_3.i_3 = float(0.0);
        }
        l_0_4.i_0 = l_0_4.i_0 * l_0_3.i_0;
        l_0_4.i_1 = l_0_4.i_1 * l_0_3.i_1;
        l_0_4.i_2 = l_0_4.i_2 * l_0_3.i_2;
        l_0_4.i_3 = l_0_4.i_3 * l_0_3.i_3;
        l_0_4.i_0 = cos(l_0_4.i_0);
        l_0_4.i_1 = cos(l_0_4.i_1);
        l_0_4.i_2 = cos(l_0_4.i_2);
        l_0_4.i_3 = cos(l_0_4.i_3);
        uint l_3_4;
        bool l_3_5;
        l_3_4 = info[(3 * 2 * info[0]) + 3] / 4;
        l_3_5 = idxGlobal < l_3_4;
        if (l_3_5) {
          l_0_3 = output_0[idxGlobal];
        } else {
          l_0_3.i_0 = float(0.0);
          l_0_3.i_1 = float(0.0);
          l_0_3.i_2 = float(0.0);
          l_0_3.i_3 = float(0.0);
        }
        l_0_3.i_0 = l_0_3.i_0 + l_0_4.i_0;
        l_0_3.i_1 = l_0_3.i_1 + l_0_4.i_1;
        l_0_3.i_2 = l_0_3.i_2 + l_0_4.i_2;
        l_0_3.i_3 = l_0_3.i_3 + l_0_4.i_3;
        uint l_3_6;
        bool l_3_7;
        l_3_6 = info[(3 * 2 * info[0]) + 3] / 4;
        l_3_7 = idxGlobal < l_3_6;
        if (l_3_7) {
          output_0[idxGlobal] = l_0_3;
        }
      }
    }
  }
}