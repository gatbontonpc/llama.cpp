
set(TARGET tasks_common)

add_library(${TARGET} STATIC
    dummy.cpp
    hellaswag.cpp
    multiple-choice.cpp
    winograde.cpp
    task-helpers.cpp
)

target_link_libraries(tasks_common common)