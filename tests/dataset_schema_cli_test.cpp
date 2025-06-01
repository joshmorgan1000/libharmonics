#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

#include <harmonics/dataset.hpp>

using namespace harmonics;

namespace {

static HTensor make_tensor(float v) {
    HTensor t{HTensor::DType::Float32, {1}};
    t.data().resize(sizeof(float));
    std::memcpy(t.data().data(), &v, sizeof(float));
    return t;
}

void create_hdf5(const char* path) {
    Hdf5Consumer cons(path);
    cons.push(make_tensor(1.f));
    cons.push(make_tensor(2.f));
}

} // namespace

TEST(DatasetSchemaCli, Valid) {
    const char* path = "schema_ok.h5";
    create_hdf5(path);
    std::string cmd = std::string("./dataset_schema_cli ") + path + " f32 1 > /dev/null";
    ASSERT_EQ(std::system(cmd.c_str()), 0);
    std::remove(path);
}

TEST(DatasetSchemaCli, WrongShape) {
    const char* path = "schema_shape.h5";
    create_hdf5(path);
    std::string cmd = std::string("./dataset_schema_cli ") + path + " f32 2 > /dev/null";
    ASSERT_EQ(std::system(cmd.c_str()) == 0, false);
    std::remove(path);
}

TEST(DatasetSchemaCli, WrongDType) {
    const char* path = "schema_dtype.h5";
    create_hdf5(path);
    std::string cmd = std::string("./dataset_schema_cli ") + path + " f64 1 > /dev/null";
    ASSERT_EQ(std::system(cmd.c_str()) == 0, false);
    std::remove(path);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
