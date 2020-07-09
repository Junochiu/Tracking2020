#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "trax" for configuration ""
set_property(TARGET trax APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(trax PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libtrax.so.3.0.2"
  IMPORTED_SONAME_NOCONFIG "libtrax.so.2"
  )

list(APPEND _IMPORT_CHECK_TARGETS trax )
list(APPEND _IMPORT_CHECK_FILES_FOR_trax "${_IMPORT_PREFIX}/lib/libtrax.so.3.0.2" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
