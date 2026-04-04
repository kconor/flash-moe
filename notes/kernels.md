- metal is just C with GPU annotations
- anotations are function parameter anotations for telling the GPU which thread is running
 - buffer(i)
 - thread group
 - thread position within group
   - lid = simd-group * 32 + simd-lane
 - simd lane: group of threads within a thread group run in lockstep 
 - threads per threadgroup


 M4 pro gpu
 - 20 cores
 - 320 execution units
 - 2560 arithmetic logic units (arithmetic and bitwise ops on integers)
 - bandwidth 273 GB/s


Basic loop
- for each output row
 - for each weight value in row
  - extract quantized integer from packed byte
  - dequantize: float-val = integer * scale + bias
  - accumulate: sum += float-val * input[col]
 - write sum to output

- architecture
 - 40 layers
 - 2048 hidden dimensions
 - 256 experts per layer
 - GatedDeltaNet + full attention
 - 
