cmake_minimum_required(VERSION 3.14)
project(cpp_bm25 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(Python_ROOT_DIR "E:/miniconda/envs/env_nlp")
set(Python_EXECUTABLE "${Python_ROOT_DIR}/python.exe")
find_package(Python REQUIRED COMPONENTS Interpreter Development)

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/pybind11-3.0.4)

pybind11_add_module(_cpp_bm25
	src/bm25_engine.cpp
	src/inverted_index.cpp
	src/tokenizer.cpp
	src/pybind_module.cpp
)

target_include_directories(_cpp_bm25 PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(WIN32)
    # On Windows pybind11_add_module already handles linking; just make sure
    # we produce a .pyd extension.
    set_target_properties(_cpp_bm25 PROPERTIES
        PREFIX ""
        SUFFIX "${Python_SOABI}.pyd"
    )
else()
    set_target_properties(_cpp_bm25 PROPERTIES
        PREFIX ""
        SUFFIX "${Python_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}"
    )
endif()

install(TARGETS _cpp_bm25
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}
    RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}
)