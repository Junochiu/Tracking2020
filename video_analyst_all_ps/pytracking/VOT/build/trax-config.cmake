#  CMake config file for the TraX library
#
#  It defines the following variables:
#    TRAX_INCLUDE_DIRS - list of include directories
#    TRAX_LIBRARIES    - list of libraries to link against
#    TRAX_LIBRARY_DIRS - list of library directories


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was trax-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

# Compute paths
get_filename_component(trax_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
set(TRAX_LIBRARIES "")

IF(NOT trax_FIND_COMPONENTS)
  SET(trax_FIND_COMPONENTS core)
ENDIF()

FOREACH(COMPONENT ${trax_FIND_COMPONENTS})
IF(NOT EXISTS ${trax_CMAKE_DIR}/trax-component-${COMPONENT}.cmake)
  SET(trax_${COMPONENT}_FOUND 0)
  IF(trax_FIND_REQUIRED_${COMPONENT})
	MESSAGE(FATAL_ERROR "TraX component ${COMPONENT} not available.")
  ENDIF()
ELSE()
  INCLUDE("${trax_CMAKE_DIR}/trax-component-${COMPONENT}.cmake")
  SET(trax_${COMPONENT}_FOUND 1)
ENDIF()
ENDFOREACH()

IF (trax_core_FOUND)
LIST(APPEND TRAX_LIBRARIES "trax")
ENDIF()

IF (trax_client_FOUND)
LIST(APPEND TRAX_LIBRARIES "trax_client")
ENDIF()

IF (trax_opencv_FOUND)
LIST(APPEND TRAX_LIBRARIES "trax_opencv")
ENDIF()

SET(TRAX_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include" "${PACKAGE_PREFIX_DIR}/include/trax")
SET(TRAX_LIBRARY_DIRS "${PACKAGE_PREFIX_DIR}/lib")

set(TRAX_FOUND TRUE)

