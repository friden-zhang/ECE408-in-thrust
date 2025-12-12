add_executable(mp3 template.cu)

target_link_libraries(
  mp3 PRIVATE
  cxxopts::cxxopts
  CUDA::cudart
)

set_target_properties(
  mp3 PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
