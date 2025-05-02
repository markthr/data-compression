include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/52eb8108c5bdec04579160ae17225d66034bd723.zip
) # using 1.17 https://github.com/google/googletest/releases/tag/v1.17.0
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(fft_test fft_test.cpp)
target_link_libraries(fft_test
    GTest::gtest_main
)


include(GoogleTest)
gtest_discover_tests(fft_test)