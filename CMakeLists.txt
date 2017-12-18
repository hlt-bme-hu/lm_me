
project(lm_me)

include_directories(src)
add_executable(lm_me src/lm_me.cpp src/Reader.h src/Reader.cpp src/Sparsity.h src/Sparsity.cpp )
