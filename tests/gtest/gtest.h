#ifndef GTEST_GTEST_H_
#define GTEST_GTEST_H_

#include <functional>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace testing {
struct Test {
    std::string name;
    std::function<void()> func;
};
inline std::vector<Test>& tests() {
    static std::vector<Test> t;
    return t;
}
inline bool registerTest(const std::string& name, std::function<void()> func) {
    tests().push_back({name, func});
    return true;
}
inline void InitGoogleTest(int*, char**) {}
inline int RunAllTests() {
    int failures = 0;
    for (auto& t : tests()) {
        try {
            t.func();
            std::cout << "[ RUN      ] " << t.name << std::endl;
            std::cout << "[       OK ] " << t.name << std::endl;
        } catch (const std::exception& e) {
            ++failures;
            std::cout << "[  FAILED  ] " << t.name << " - " << e.what() << std::endl;
        } catch (...) {
            ++failures;
            std::cout << "[  FAILED  ] " << t.name << " - unknown exception" << std::endl;
        }
    }
    if (failures)
        std::cout << failures << " test(s) failed" << std::endl;
    else
        std::cout << "All tests passed" << std::endl;
    return failures == 0 ? 0 : 1;
}

namespace internal {
class TestFailureException : public std::runtime_error {
  public:
    explicit TestFailureException(const std::string& msg) : std::runtime_error(msg) {}
};
} // namespace internal
} // namespace testing

#define RUN_ALL_TESTS() ::testing::RunAllTests()

#define EXPECT_EQ(val1, val2)                                                                      \
    do {                                                                                           \
        if (!((val1) == (val2))) {                                                                 \
            throw ::testing::internal::TestFailureException(std::string("EXPECT_EQ failed: ") +    \
                                                            #val1 " != " #val2);                   \
        }                                                                                          \
    } while (0)

#define ASSERT_EQ(val1, val2) \
    do { \
        if (!((val1) == (val2))) { \
            throw ::testing::internal::TestFailureException(std::string("ASSERT_EQ failed: ") + \
                                                            #val1 " != " #val2); \
        } \
    } while (0)

#define EXPECT_FLOAT_EQ(val1, val2) \
    do { \
        if (std::fabs((val1) - (val2)) > 1e-5f) { \
            throw ::testing::internal::TestFailureException(std::string("EXPECT_FLOAT_EQ failed: ") + \
                                                            #val1 " != " #val2); \
        } \
    } while (0)

#define TEST(suite, name)                                                                          \
    static void suite##_##name##_func();                                                           \
    static bool suite##_##name##_registered =                                                      \
        ::testing::registerTest(#suite "/" #name, suite##_##name##_func);                          \
    static void suite##_##name##_func()

#endif // GTEST_GTEST_H_
