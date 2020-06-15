include(CMakePackageConfigHelpers)

#
# Config
#
configure_package_config_file(LibraryConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
	INSTALL_DESTINATION .
	)

#
# Targets
#
install(
	EXPORT Targets
	FILE ${PROJECT_NAME}Targets.cmake
	NAMESPACE ${PROJECT_PREFIX}::${MODULE_PREFIX}::
	DESTINATION cmake
	)

install(
	FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
	DESTINATION cmake
	)