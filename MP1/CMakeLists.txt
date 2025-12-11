add_executable(mp1 template.cu)

target_link_libraries(
  mp1 PRIVATE
  cxxopts::cxxopts
  CUDA::cudart
)

set_target_properties(
  mp1 PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
