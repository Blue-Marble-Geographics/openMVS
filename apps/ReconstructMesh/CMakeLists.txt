if(MSVC)
	FILE(GLOB LIBRARY_FILES_C "*.cpp" "*.rc")
else()
	FILE(GLOB LIBRARY_FILES_C "*.cpp")
endif()
FILE(GLOB LIBRARY_FILES_H "*.h" "*.inl")

find_package(TBB CONFIG REQUIRED)
cxx_executable_with_flags(ReconstructMesh "Apps" "${cxx_default}" "MVS;${OpenMVS_EXTRA_LIBS};TBB::tbb" ${LIBRARY_FILES_C} ${LIBRARY_FILES_H})

# Install
INSTALL(TARGETS ReconstructMesh
	EXPORT OpenMVSTargets
	RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin)
