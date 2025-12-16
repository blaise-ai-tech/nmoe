#pragma once

#include <string>
#include <exception>

#include "configs.cuh"

#ifndef RDEP_STATIC_ASSERT
#define RDEP_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

class RDEPException: public std::exception {
private:
    std::string message = {};

public:
    explicit RDEPException(const char *name, const char* file, const int line, const std::string& error) {
        message = std::string("Failed: ") + name + " error " + file + ":" + std::to_string(line) + " '" + error + "'";
    }

    const char *what() const noexcept override { return message.c_str(); }
};

#ifndef RDEP_CUDA_CHECK
#define RDEP_CUDA_CHECK(cmd) \
do { \
    cudaError_t e = (cmd); \
    if (e != cudaSuccess) { \
        throw RDEPException("CUDA", __FILE__, __LINE__, cudaGetErrorString(e)); \
    } \
} while (0)
#endif

#ifndef RDEP_HOST_ASSERT
#define RDEP_HOST_ASSERT(cond) \
do { \
    if (not (cond)) { \
        throw RDEPException("Assertion", __FILE__, __LINE__, #cond); \
    } \
} while (0)
#endif

#ifndef RDEP_DEVICE_ASSERT
#define RDEP_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) { \
        printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
        asm("trap;"); \
    } \
} while (0)
#endif
