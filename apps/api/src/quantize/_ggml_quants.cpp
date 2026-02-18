#define PY_SSIZE_T_CLEAN
#include <Python.h>
#
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <exception>
#include <thread>
#include <cstdlib>
#include <climits>
#include <vector>
#
// Minimal ggml ABI surface (ggml 0.9.5+) for quant/dequant.
extern "C" {
typedef void (*ggml_to_float_t)(const void * x, float * y, int64_t k);
typedef void (*ggml_from_float_t)(const float * x, void * y, int64_t k);
struct ggml_type_traits {
    const char * type_name;
    int64_t blck_size;
    int64_t blck_size_interleave;
    size_t type_size;
    bool is_quantized;
    ggml_to_float_t to_float;
    ggml_from_float_t from_float_ref;
};

const struct ggml_type_traits * ggml_get_type_traits(int type);
bool ggml_quantize_requires_imatrix(int type);
size_t ggml_quantize_chunk(int type, const float * src, void * dst, int64_t start,
                           int64_t nrows, int64_t n_per_row, const float * imatrix);
}  // extern "C"
#
namespace {
int ggml_quants_num_threads(int64_t nrows, int64_t n_per_row) {
    const char * env = std::getenv("APEX_GGML_QUANTS_THREADS");
    long configured = 0;  // 0 => auto
    bool explicit_threads = false;
    if (env != nullptr && env[0] != '\0') {
        char * end = nullptr;
        configured = std::strtol(env, &end, 10);
        if (end == env) {
            configured = -1;  // invalid -> fall back to 1 thread
        }
        explicit_threads = configured > 0;
    }

    const unsigned hw = std::thread::hardware_concurrency();
    int max_threads = hw ? int(hw) : 256;
    if (max_threads > 256) {
        max_threads = 256;
    }

    int threads = 1;
    if (configured > 0) {
        threads = configured > max_threads ? max_threads : int(configured);
    } else if (configured == 0) {
        threads = hw ? int(hw) : 1;
    } else {
        threads = 1;
    }

    if (threads < 1) {
        threads = 1;
    }
    if (nrows > 0 && int64_t(threads) > nrows) {
        threads = nrows > INT_MAX ? INT_MAX : int(nrows);
    }

    // Avoid thread spawn overhead on small workloads. Threshold is in float
    // elements (~4 MiB of input).
    const __int128 total = (__int128)nrows * (__int128)n_per_row;
    if (!explicit_threads && threads > 1 && total < (__int128(1) << 20)) {
        threads = 1;
    }

    return threads;
}

PyObject * py_quantize(PyObject * /*self*/, PyObject * args) {
    PyObject * arr_obj = nullptr;
    int type = 0;
    if (!PyArg_ParseTuple(args, "Oi", &arr_obj, &type)) {
        return nullptr;
    }

    if (ggml_quantize_requires_imatrix(type)) {
        PyErr_SetString(PyExc_NotImplementedError,
                        "ggml quantization for this type requires an importance matrix");
        return nullptr;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(type);
    if (traits == nullptr) {
        PyErr_SetString(PyExc_ValueError, "Unknown ggml type");
        return nullptr;
    }

    PyArrayObject * in =
        (PyArrayObject *)PyArray_FROM_OTF(arr_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (in == nullptr) {
        return nullptr;
    }

    const int ndim = PyArray_NDIM(in);
    if (ndim < 1) {
        Py_DECREF(in);
        PyErr_SetString(PyExc_ValueError, "Input must have at least 1 dimension");
        return nullptr;
    }

    const npy_intp * in_dims = PyArray_DIMS(in);
    const int64_t n_per_row = int64_t(in_dims[ndim - 1]);

    if (traits->blck_size == 0 || (n_per_row % traits->blck_size) != 0) {
        Py_DECREF(in);
        PyErr_SetString(PyExc_ValueError, "Last dimension is not divisible by block size");
        return nullptr;
    }

    const int64_t out_row_bytes =
        (n_per_row / traits->blck_size) * int64_t(traits->type_size);

    std::vector<npy_intp> out_dims(ndim);
    for (int i = 0; i < ndim; ++i) {
        out_dims[i] = in_dims[i];
    }
    out_dims[ndim - 1] = (npy_intp)out_row_bytes;

    PyArrayObject * out =
        (PyArrayObject *)PyArray_SimpleNew(ndim, out_dims.data(), NPY_UINT8);
    if (out == nullptr) {
        Py_DECREF(in);
        return nullptr;
    }

    const int64_t nrows = (n_per_row == 0) ? 0 : (PyArray_SIZE(in) / n_per_row);
    const float * src = (const float *)PyArray_DATA(in);
    void * dst = (void *)PyArray_DATA(out);

    const int threads = ggml_quants_num_threads(nrows, n_per_row);

    bool failed = false;
    char error[256] = {0};
    std::vector<std::thread> workers;

    Py_BEGIN_ALLOW_THREADS
    try {
        if (threads <= 1 || nrows <= 1) {
            (void)ggml_quantize_chunk(type, src, dst, 0, nrows, n_per_row, nullptr);
        } else {
            workers.reserve(size_t(threads));

            const int64_t rows_per = nrows / threads;
            const int64_t remainder = nrows % threads;
            int64_t start_row = 0;
            for (int i = 0; i < threads; ++i) {
                const int64_t take = rows_per + (i < remainder ? 1 : 0);
                const int64_t s = start_row;
                start_row += take;
                if (take <= 0) {
                    continue;
                }
                workers.emplace_back([=]() {
                    (void)ggml_quantize_chunk(type, src, dst, s * n_per_row, take, n_per_row, nullptr);
                });
            }
            for (auto & t : workers) {
                t.join();
            }
        }
    } catch (const std::exception & e) {
        failed = true;
        std::snprintf(error, sizeof(error), "%s", e.what());
        for (auto & t : workers) {
            if (t.joinable()) {
                t.join();
            }
        }
    } catch (...) {
        failed = true;
        std::snprintf(error, sizeof(error), "unknown error");
        for (auto & t : workers) {
            if (t.joinable()) {
                t.join();
            }
        }
    }
    Py_END_ALLOW_THREADS
    if (failed) {
        Py_DECREF(in);
        Py_DECREF(out);
        PyErr_SetString(PyExc_RuntimeError, error[0] ? error : "ggml quantize failed");
        return nullptr;
    }

    Py_DECREF(in);
    return (PyObject *)out;
}

PyObject * py_dequantize(PyObject * /*self*/, PyObject * args) {
    PyObject * arr_obj = nullptr;
    int type = 0;
    if (!PyArg_ParseTuple(args, "Oi", &arr_obj, &type)) {
        return nullptr;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(type);
    if (traits == nullptr || traits->to_float == nullptr) {
        PyErr_SetString(PyExc_ValueError, "Unknown ggml type (or no dequantizer)");
        return nullptr;
    }

    PyArrayObject * in =
        (PyArrayObject *)PyArray_FROM_OTF(arr_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    if (in == nullptr) {
        return nullptr;
    }

    const int ndim = PyArray_NDIM(in);
    if (ndim < 1) {
        Py_DECREF(in);
        PyErr_SetString(PyExc_ValueError, "Input must have at least 1 dimension");
        return nullptr;
    }

    const npy_intp * in_dims = PyArray_DIMS(in);
    const int64_t row_bytes = int64_t(in_dims[ndim - 1]);

    if (traits->type_size == 0 || (row_bytes % int64_t(traits->type_size)) != 0) {
        Py_DECREF(in);
        PyErr_SetString(PyExc_ValueError, "Last dimension is not divisible by type size");
        return nullptr;
    }

    const int64_t n_per_row =
        (row_bytes / int64_t(traits->type_size)) * int64_t(traits->blck_size);

    std::vector<npy_intp> out_dims(ndim);
    for (int i = 0; i < ndim; ++i) {
        out_dims[i] = in_dims[i];
    }
    out_dims[ndim - 1] = (npy_intp)n_per_row;

    PyArrayObject * out =
        (PyArrayObject *)PyArray_SimpleNew(ndim, out_dims.data(), NPY_FLOAT32);
    if (out == nullptr) {
        Py_DECREF(in);
        return nullptr;
    }

    const int64_t nrows = (row_bytes == 0) ? 0 : (PyArray_SIZE(in) / row_bytes);
    const uint8_t * src = (const uint8_t *)PyArray_DATA(in);
    float * dst = (float *)PyArray_DATA(out);

    const int threads = ggml_quants_num_threads(nrows, n_per_row);

    bool failed = false;
    char error[256] = {0};
    std::vector<std::thread> workers;

    Py_BEGIN_ALLOW_THREADS
    try {
        if (threads <= 1 || nrows <= 1) {
            for (int64_t r = 0; r < nrows; ++r) {
                traits->to_float((const void *)(src + r * row_bytes), dst + r * n_per_row, n_per_row);
            }
        } else {
            workers.reserve(size_t(threads));

            const int64_t rows_per = nrows / threads;
            const int64_t remainder = nrows % threads;
            int64_t start_row = 0;
            for (int i = 0; i < threads; ++i) {
                const int64_t take = rows_per + (i < remainder ? 1 : 0);
                const int64_t s = start_row;
                const int64_t e = s + take;
                start_row = e;
                if (take <= 0) {
                    continue;
                }

                workers.emplace_back([=]() {
                    for (int64_t r = s; r < e; ++r) {
                        traits->to_float((const void *)(src + r * row_bytes), dst + r * n_per_row, n_per_row);
                    }
                });
            }
            for (auto & t : workers) {
                t.join();
            }
        }
    } catch (const std::exception & e) {
        failed = true;
        std::snprintf(error, sizeof(error), "%s", e.what());
        for (auto & t : workers) {
            if (t.joinable()) {
                t.join();
            }
        }
    } catch (...) {
        failed = true;
        std::snprintf(error, sizeof(error), "unknown error");
        for (auto & t : workers) {
            if (t.joinable()) {
                t.join();
            }
        }
    }
    Py_END_ALLOW_THREADS
    if (failed) {
        Py_DECREF(in);
        Py_DECREF(out);
        PyErr_SetString(PyExc_RuntimeError, error[0] ? error : "ggml dequantize failed");
        return nullptr;
    }

    Py_DECREF(in);
    return (PyObject *)out;
}

static PyMethodDef methods[] = {
    {"quantize", (PyCFunction)py_quantize, METH_VARARGS,
     "quantize(x: np.ndarray[float32], qtype: int) -> np.ndarray[uint8]"},
    {"dequantize", (PyCFunction)py_dequantize, METH_VARARGS,
     "dequantize(x: np.ndarray[uint8], qtype: int) -> np.ndarray[float32]"},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_ggml_quants",
    "Fast ggml-backed quantize/dequantize helpers.",
    -1,
    methods,
};
}  // namespace

PyMODINIT_FUNC PyInit__ggml_quants(void) {
    import_array();
    return PyModule_Create(&module_def);
}
