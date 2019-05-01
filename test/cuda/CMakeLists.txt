aux_source_directory(. DIR_SRCS)
add_executable(unit_tests ${DIR_SRCS})
target_link_libraries(unit_tests cu gtest_main)
add_test(NAME units_tests COMMAND unit_tests)