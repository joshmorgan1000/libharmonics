#ifndef GTEST_GTEST_H_
#define GTEST_GTEST_H_

#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <typeinfo>
#include <type_traits>
#include <utility>

// Define std::void_t for pre-C++17 compatibility
#if __cplusplus < 201703L
namespace std {
    template<typename...>
    using void_t = void;
}
#endif

namespace testing {

template <typename, typename, typename = void>
struct is_equality_comparable : std::false_type {};

template <typename A, typename B>
struct is_equality_comparable<A, B,
    std::void_t<decltype(std::declval<A>() == std::declval<B>())>> : std::true_type {};


struct SkipException : std::exception {
    explicit SkipException(std::string  m) : msg_(std::move(m)) {}
    const char* what() const noexcept override { return msg_.c_str(); }
private:
    std::string msg_;
};

struct TestCase {
    const char* suite;
    const char* name;
    void (*func)();
};

inline bool g_skipped = false; 

inline std::vector<TestCase>& GetTests() {
    static std::vector<TestCase> tests;
    return tests;
}

inline void RegisterTest(const char* suite, const char* name, void (*func)()) {
    GetTests().push_back({suite, name, func});
}

class AssertionFailure : public std::exception {
  public:
    explicit AssertionFailure(std::string msg) : msg_(std::move(msg)) {}
    const char* what() const noexcept override { return msg_.c_str(); }

  private:
    std::string msg_;
};

inline int g_failures = 0;

inline void ReportFailure(const char* msg) {
    std::cerr << msg << std::endl;
    ++g_failures;
}

inline void ExpectTrue(bool condition, const char* file, int line,
                          const char* expr, bool fatal) {
    if (!condition) {
        std::ostringstream oss;
        oss << file << ":" << line << ": EXPECT_TRUE failed:\n  " << expr;
        ReportFailure(oss.str().c_str());
        if (fatal) throw AssertionFailure(oss.str());
    }
}

inline void ExpectFalse(bool condition, const char* file, int line,
                       const char* expr, bool fatal) {
    if (condition) {
        std::ostringstream oss;
        oss << file << ":" << line << ": EXPECT_FALSE failed:\n  " << expr;
        ReportFailure(oss.str().c_str());
        if (fatal) throw AssertionFailure(oss.str());
    }
}

template <class A, class B>
inline void ExpectEq(const A& a, const B& b, const char* file, int line,
                     const char* expr_a, const char* expr_b, bool fatal) {
    // Check if they are not comparable object types, if not fail the test
    if constexpr (!is_equality_comparable<A, B>::value) {
        std::ostringstream oss;
        oss << file << ":" << line << ": EXPECT_EQ failed:\n  " << expr_a
            << " vs " << expr_b << " (types not comparable)";
        ReportFailure(oss.str().c_str());
        if (fatal) throw AssertionFailure(oss.str());
        return;
    }
    if (!(a == b)) {
        std::ostringstream oss;
        oss << file << ":" << line << ": EXPECT_EQ failed:\n  " << expr_a
            << " vs " << expr_b;
        ReportFailure(oss.str().c_str());
        if (fatal) throw AssertionFailure(oss.str());
    }
}

inline void ExpectFloatEq(float a, float b, const char* file, int line,
                          const char* expr_a, const char* expr_b, bool fatal) {
    if (std::fabs(a - b) > 1e-5f) {
        std::ostringstream oss;
        oss << file << ":" << line << ": EXPECT_FLOAT_EQ failed:\n  " << expr_a
            << " vs " << expr_b << " (" << a << " vs " << b << ")";
        ReportFailure(oss.str().c_str());
        if (fatal) throw AssertionFailure(oss.str());
    }
}

inline void ExpectFloatNear(float a, float b, float tol, const char* file,
                            int line, const char* expr_a, const char* expr_b,
                            bool fatal) {
    if (std::fabs(a - b) > tol) {
        std::ostringstream oss;
        oss << file << ":" << line << ": EXPECT_NEAR failed:\n  " << expr_a
            << " vs " << expr_b << " (" << a << " vs " << b
            << ", tol=" << tol << ")";
        ReportFailure(oss.str().c_str());
        if (fatal) throw AssertionFailure(oss.str());
    }
}

template <class A, class B>
inline void ExpectNe(const A& a, const B& b, const char* file, int line,
                     const char* expr_a, const char* expr_b, bool fatal) {
    if (!(a != b)) {
        std::ostringstream oss;
        oss << file << ":" << line << ": EXPECT_NE failed:\n  " << expr_a
            << " vs " << expr_b;
        ReportFailure(oss.str().c_str());
        if (fatal) throw AssertionFailure(oss.str());
    }
}

template <class A, class B>
inline void ExpectLessThan(const A& a, const B& b, const char* file, int line,
                             const char* expr_a, const char* expr_b, bool fatal) {
    if (!(a < b)) {
        std::ostringstream oss;
        oss << file << ":" << line << ": EXPECT_LT failed:\n  " << expr_a
            << " vs " << expr_b;
        ReportFailure(oss.str().c_str());
        if (fatal) throw AssertionFailure(oss.str());
    }
}
template <class A, class B>
inline void ExpectGreaterThan(const A& a, const B& b, const char* file, int line,
                             const char* expr_a, const char* expr_b, bool fatal) {
    if (!(a > b)) {
        std::ostringstream oss;
        oss << file << ":" << line << ": EXPECT_GT failed:\n  " << expr_a
            << " vs " << expr_b;
        ReportFailure(oss.str().c_str());
        if (fatal) throw AssertionFailure(oss.str());
    }
}

struct SkipBuilder {
    SkipBuilder(const char* file, int line) : file_(file), line_(line) {}
    SkipBuilder(SkipBuilder&& other) noexcept
        : file_(other.file_), line_(other.line_), oss_(std::move(other.oss_)) {}
    SkipBuilder(const SkipBuilder&)            = delete;
    SkipBuilder& operator=(const SkipBuilder&) = delete;
    SkipBuilder& operator=(SkipBuilder&&)      = delete;
    template <class T>
    SkipBuilder& operator<<(const T& v) & {
        oss_ << v;
        return *this;
    }
    template <class T>
    SkipBuilder&& operator<<(const T& v) && {
        oss_ << v;
        return std::move(*this);
    }
    operator SkipException() const {
        std::ostringstream out;
        out << file_ << ":" << line_
            << (oss_.str().empty() ? "" : " ") << oss_.str();
        return SkipException(out.str());
    }

private:
    const char*        file_;
    int                line_;
    std::ostringstream oss_;
};

inline void InitGoogleTest(int*, char**) {}

inline int RunAllTests() {
    int failed_tests   = 0;
    int skipped_tests  = 0;

    for (auto& t : GetTests()) {
        g_failures = 0;
        std::cout << "[ RUN      ] " << t.suite << "." << t.name << '\n';
        try {
            t.func();
            if (g_failures == 0) {
                std::cout << "[       OK ] " << t.suite << "." << t.name << '\n';
            } else {
                std::cout << "[  FAILED  ] " << t.suite << "." << t.name << '\n';
                ++failed_tests;
            }
        } catch (const ::testing::SkipBuilder&) {
            std::cout << "[  SKIPPED ] " << t.suite << "." << t.name << '\n';
            ++skipped_tests;
        } catch (const SkipException& e) {
            std::cout << "[  SKIPPED ] " << t.suite << "." << t.name
                      << " " << e.what() << '\n';
            ++skipped_tests;
        } catch (const std::exception& e) {
            std::cout << "[  FAILED  ] " << t.suite << "." << t.name
                      << " unexpected exception: " << e.what() << '\n';
            ++failed_tests;
        } catch (...) {
            std::cout << "[  FAILED  ] " << t.suite << "." << t.name
                      << " unknown exception\n";
            ++failed_tests;
        }
    }

    std::cout << "[  PASSED  ] " << (GetTests().size() - failed_tests - skipped_tests)
              << " tests\n"
              << "[  SKIPPED ] " << skipped_tests << " tests\n";
    if (failed_tests) std::cout << "[  FAILED  ] " << failed_tests << " tests\n";

    return failed_tests == 0 ? 0 : 1;
}

namespace internal {

namespace  {

// stdout state
inline std::ostringstream  g_cout_stream;
inline std::streambuf*     g_saved_cout_buf   = nullptr;
inline bool                g_cout_capturing   = false;

// stderr state
inline std::ostringstream  g_cerr_stream;
inline std::streambuf*     g_saved_cerr_buf   = nullptr;
inline bool                g_cerr_capturing   = false;

} // unnamed namespace

// ----- stdout --------------------------------------------------------------
inline void CaptureStdout() {
    if (g_cout_capturing)
        throw std::runtime_error("CaptureStdout() already active");
    g_cout_stream.str({});
    g_cout_stream.clear();
    g_saved_cout_buf = std::cout.rdbuf(g_cout_stream.rdbuf());
    g_cout_capturing = true;
}

inline std::string GetCapturedStdout() {
    if (!g_cout_capturing)
        throw std::runtime_error("GetCapturedStdout() without CaptureStdout()");
    std::cout.rdbuf(g_saved_cout_buf);
    g_cout_capturing = false;
    return g_cout_stream.str();
}

// ----- stderr --------------------------------------------------------------
inline void CaptureStderr() {
    if (g_cerr_capturing)
        throw std::runtime_error("CaptureStderr() already active");
    g_cerr_stream.str({});
    g_cerr_stream.clear();
    g_saved_cerr_buf = std::cerr.rdbuf(g_cerr_stream.rdbuf());
    g_cerr_capturing = true;
}

inline std::string GetCapturedStderr() {
    if (!g_cerr_capturing)
        throw std::runtime_error("GetCapturedStderr() without CaptureStderr()");
    std::cerr.rdbuf(g_saved_cerr_buf);
    g_cerr_capturing = false;
    return g_cerr_stream.str();
}

} // namespace internal

} // namespace testing

#define TEST(suite, name)                                                     \
    static void suite##_##name##_Test();                                      \
    namespace {                                                               \
    struct suite##_##name##_Reg {                                             \
        suite##_##name##_Reg() {                                              \
            ::testing::RegisterTest(#suite, #name, &suite##_##name##_Test);   \
        }                                                                     \
    };                                                                        \
    static suite##_##name##_Reg suite##_##name##_Reg_instance;                \
    } /* anonymous namespace */                                               \
    static void suite##_##name##_Test()
#define EXPECT_EQ(a, b)                                                       \
    ::testing::ExpectEq((a), (b), __FILE__, __LINE__, #a, #b, false)
#define ASSERT_EQ(a, b)                                                       \
    ::testing::ExpectEq((a), (b), __FILE__, __LINE__, #a, #b, true)
#define EXPECT_FLOAT_EQ(a, b)                                                 \
    ::testing::ExpectFloatEq((a), (b), __FILE__, __LINE__, #a, #b, false)
#define ASSERT_FLOAT_EQ(a, b)                                                 \
    ::testing::ExpectFloatEq((a), (b), __FILE__, __LINE__, #a, #b, true)
#define EXPECT_LT(a, b)                                                       \
    ::testing::ExpectLessThan((a), (b), __FILE__, __LINE__, #a, #b, false)
#define ASSERT_LT(a, b)                                                       \
    ::testing::ExpectLessThan((a), (b), __FILE__, __LINE__, #a, #b, true)
#define EXPECT_GT(a, b)                                                       \
    ::testing::ExpectGreaterThan((a), (b), __FILE__, __LINE__, #a, #b, false)
#define ASSERT_GT(a, b)                                                       \
    ::testing::ExpectGreaterThan((a), (b), __FILE__, __LINE__, #a, #b, true)
#define EXPECT_NEAR(a, b, tol)                                                \
    ::testing::ExpectFloatNear((a), (b), (tol), __FILE__, __LINE__, #a, #b, false)
#define ASSERT_NEAR(a, b, tol)                                                \
    ::testing::ExpectFloatNear((a), (b), (tol), __FILE__, __LINE__, #a, #b, true)
#define RUN_ALL_TESTS()                                                       \
    ::testing::RunAllTests()
#define EXPECT_NE(a, b)                                                       \
    ::testing::ExpectEq((a), (b), __FILE__, __LINE__, #a, #b, false)
#define ASSERT_NE(a, b)                                                       \
    ::testing::ExpectNe((a), (b), __FILE__, __LINE__, #a, #b, true)
#define EXPECT_TRUE(a)                                                        \
    ::testing::ExpectTrue((a), __FILE__, __LINE__, #a, false)
#define ASSERT_TRUE(a)                                                        \
    ::testing::ExpectTrue((a), __FILE__, __LINE__, #a, true)
#define EXPECT_FALSE(a)                                                       \
    ::testing::ExpectFalse((a), __FILE__, __LINE__, #a, false)
#define ASSERT_FALSE(a)                                                       \
    ::testing::ExpectFalse((a), __FILE__, __LINE__, #a, true)
#define GTEST_SKIP()  throw ::testing::SkipBuilder(__FILE__, __LINE__)

#endif // GTEST_GTEST_H_
