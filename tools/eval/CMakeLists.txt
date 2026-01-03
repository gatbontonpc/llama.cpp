add_subdirectory(tasks)

set(TARGET llama-eval)
add_executable(${TARGET} llama-eval.cpp)
target_link_libraries(${TARGET} PRIVATE llama tasks_common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

