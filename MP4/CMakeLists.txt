add_executable(mp4 template.cu)

target_link_libraries(
  mp4 PRIVATE
  cxxopts::cxxopts
  CUDA::cudart
)

set_target_properties(
  mp4 PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
