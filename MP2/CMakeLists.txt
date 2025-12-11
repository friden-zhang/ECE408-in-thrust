add_executable(mp2 template.cu)

target_link_libraries(
  mp2 PRIVATE
  cxxopts::cxxopts
  CUDA::cudart
)

set_target_properties(
  mp2 PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
