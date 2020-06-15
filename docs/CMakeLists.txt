#
# Detect doxygen
#
find_package(Doxygen OPTIONAL_COMPONENTS dot )
if (NOT DOXYGEN_FOUND)
    message("-- Doxygen not found. Skipping documentation.")
    return()
endif()

#
# Decide on using mathjax or latex for formula parsing
# Only needed if formulas are use in doxygen
#
if (NOT MATHJAX_PATH)
    set(MATHJAX_PATH $ENV{MATHJAX_PATH})
endif()
if (MATHJAX_PATH AND EXISTS "${MATHJAX_PATH}")
    set(USE_MATHJAX YES)
    message("-- Using mathjax for math formulas in doxygen.")
    if (NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/reference/mathjax)
        message("-- Inital copying mathjax to build folder. Please wait.")
        file(COPY "${MATHJAX_PATH}/" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/reference/mathjax)
    endif()
else()
    find_package(LATEX)
    if (NOT LATEX_FOUND)
        message("-- Latex not found. Skipping documentation.")
        return()
    endif()
    message("-- Using latex for math formulas in doxygen.")
endif()

#
# Settings
#
set(PROJECT_LOGO ${CMAKE_CURRENT_SOURCE_DIR}/logo/APSW.png)
set(OUTPUT_LANGUAGE English)


#
# Everything configure. Can go to subdirecty
#
add_subdirectory(reference)
