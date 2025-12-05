add_executable(template template.cu)

target_link_libraries(
  template PRIVATE
  cxxopts::cxxopts
  CUDA::cudart
)

set_target_properties(
  template PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
