add_subdirectory(memory)
add_subdirectory(platform)
add_subdirectory(framework)
add_subdirectory(operators)
add_subdirectory(string)

add_subdirectory(pybind)
if (NOT WIN32)
add_subdirectory(recordio)
endif(NOT WIN32)

# NOTE: please add subdirectory inference at last.
add_subdirectory(inference)

if (NOT WITH_AMD_GPU)
  add_subdirectory(train)
endif (NOT WITH_AMD_GPU)
