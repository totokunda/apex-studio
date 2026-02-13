cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0091 NEW) # MSVC runtime library selection
project(native_decoder LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# cmake-js provides these variables
include_directories(${CMAKE_JS_INC})

# ---------------------------------------------------------------------------
# FFmpeg: resolve platform-specific pre-built libraries
# ---------------------------------------------------------------------------
if(WIN32)
    set(FFMPEG_DIR "${CMAKE_SOURCE_DIR}/deps/ffmpeg/win-x64")
elseif(APPLE)
    # Resolve against the target arch (not host arch) so cross-builds on macOS work.
    set(TARGET_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
    if(CMAKE_OSX_ARCHITECTURES)
        list(LENGTH CMAKE_OSX_ARCHITECTURES OSX_ARCH_COUNT)
        if(OSX_ARCH_COUNT GREATER 1)
            message(FATAL_ERROR
                "Universal macOS build is not supported in a single pass. "
                "Build each arch separately (arm64 and x64).")
        endif()
        list(GET CMAKE_OSX_ARCHITECTURES 0 TARGET_ARCH)
    endif()
    string(TOLOWER "${TARGET_ARCH}" TARGET_ARCH)

    if(TARGET_ARCH STREQUAL "arm64" OR TARGET_ARCH STREQUAL "aarch64")
        set(FFMPEG_DIR "${CMAKE_SOURCE_DIR}/deps/ffmpeg/darwin-arm64")
    elseif(TARGET_ARCH STREQUAL "x86_64" OR TARGET_ARCH STREQUAL "amd64" OR TARGET_ARCH STREQUAL "x64")
        set(FFMPEG_DIR "${CMAKE_SOURCE_DIR}/deps/ffmpeg/darwin-x64")
    else()
        message(FATAL_ERROR "Unsupported macOS target arch: ${TARGET_ARCH}")
    endif()
else()
    set(FFMPEG_DIR "${CMAKE_SOURCE_DIR}/deps/ffmpeg/linux-x64")
endif()

if(NOT EXISTS "${FFMPEG_DIR}/include")
    message(FATAL_ERROR
        "FFmpeg headers not found at ${FFMPEG_DIR}/include. "
        "Run 'node scripts/fetch-ffmpeg.js' first "
        "(or set FFMPEG_ARCH=arm64/x64 on macOS).")
endif()

include_directories("${FFMPEG_DIR}/include")
link_directories("${FFMPEG_DIR}/lib")

# ---------------------------------------------------------------------------
# node-addon-api headers
# ---------------------------------------------------------------------------
execute_process(
    COMMAND node -p "require('node-addon-api').include"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE NODE_ADDON_API_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "\"" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})
include_directories(${NODE_ADDON_API_DIR})

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
set(SOURCES
    src/addon.cpp
    src/decoder_instance.cpp
    src/hwaccel.cpp
    src/frame_pool.cpp
    src/alpha_merger.cpp
    src/thread_pool.cpp
    src/seek_engine.cpp
)
if(APPLE)
    list(APPEND SOURCES src/apple_vt_reader.mm)
endif()

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${CMAKE_JS_SRC})

# N-API version & disable deprecated APIs
target_compile_definitions(${PROJECT_NAME} PRIVATE
    NAPI_VERSION=8
    NODE_ADDON_API_DISABLE_DEPRECATED
)

# ---------------------------------------------------------------------------
# Link FFmpeg libraries
# ---------------------------------------------------------------------------
set(FFMPEG_LIBS avformat avcodec swscale swresample avutil)

if(WIN32)
    # FFmpeg import libs on Windows need these system libraries
    foreach(lib ${FFMPEG_LIBS})
        target_link_libraries(${PROJECT_NAME} PRIVATE
            "${FFMPEG_DIR}/lib/${lib}.lib"
        )
    endforeach()
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${CMAKE_JS_LIB}
        bcrypt mfplat mfuuid strmiids ole32 user32 advapi32
        secur32 ws2_32 d3d11 dxgi
    )
elseif(APPLE)
    foreach(lib ${FFMPEG_LIBS})
        target_link_libraries(${PROJECT_NAME} PRIVATE
            "${FFMPEG_DIR}/lib/lib${lib}.dylib"
        )
    endforeach()
    # Keep runtime lookup local first when FFmpeg dylibs are copied next to the addon.
    target_link_options(${PROJECT_NAME} PRIVATE "-Wl,-rpath,@loader_path")
    find_library(COREMEDIA CoreMedia REQUIRED)
    find_library(COREVIDEO CoreVideo REQUIRED)
    find_library(VIDEOTOOLBOX VideoToolbox REQUIRED)
    find_library(COREFOUNDATION CoreFoundation REQUIRED)
    find_library(SECURITY Security REQUIRED)
    find_library(AUDIOTOOLBOX AudioToolbox REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${CMAKE_JS_LIB}
        ${COREMEDIA} ${COREVIDEO} ${VIDEOTOOLBOX}
        ${COREFOUNDATION} ${SECURITY} ${AUDIOTOOLBOX}
        iconv z bz2
    )
else()
    # Linux — link against shared libraries (.so) from BtbN shared build
    foreach(lib ${FFMPEG_LIBS})
        target_link_libraries(${PROJECT_NAME} PRIVATE
            "${FFMPEG_DIR}/lib/lib${lib}.so"
        )
    endforeach()
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${CMAKE_JS_LIB}
        pthread z dl m
    )
endif()

# ---------------------------------------------------------------------------
# Output configuration
# ---------------------------------------------------------------------------
set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".node"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build/Release"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build/Release"
)

# MSVC-specific flags
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    target_compile_options(${PROJECT_NAME} PRIVATE /W3 /O2)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -O2)
    # Enable SIMD for alpha merger
    set(SIMD_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
    if(APPLE AND DEFINED TARGET_ARCH)
        set(SIMD_ARCH "${TARGET_ARCH}")
    endif()
    string(TOLOWER "${SIMD_ARCH}" SIMD_ARCH)
    if(SIMD_ARCH MATCHES "x86_64|amd64|x64")
        target_compile_options(${PROJECT_NAME} PRIVATE -msse2 -msse4.1)
    elseif(SIMD_ARCH MATCHES "aarch64|arm64")
        # NEON is enabled by default on ARM64
    endif()
endif()

# ---------------------------------------------------------------------------
# Post-build: copy FFmpeg shared libraries alongside the .node addon
# ---------------------------------------------------------------------------
if(WIN32)
    # Copy DLLs from deps/ffmpeg/win-x64/bin/ to build/Release/
    file(GLOB FFMPEG_DLLS "${FFMPEG_DIR}/bin/*.dll")
    foreach(dll ${FFMPEG_DLLS})
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${dll}"
                "${CMAKE_SOURCE_DIR}/build/Release/"
        )
    endforeach()
elseif(APPLE)
    # macOS: copy FFmpeg dylibs alongside the addon for local runtime resolution.
    file(GLOB FFMPEG_DYLIBS "${FFMPEG_DIR}/lib/lib*.dylib*")
    foreach(dylib ${FFMPEG_DYLIBS})
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${dylib}"
                "${CMAKE_SOURCE_DIR}/build/Release/"
        )
    endforeach()
elseif(NOT APPLE)
    # Linux: copy .so files alongside the addon
    file(GLOB FFMPEG_SOS "${FFMPEG_DIR}/lib/lib*.so*")
    foreach(so ${FFMPEG_SOS})
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${so}"
                "${CMAKE_SOURCE_DIR}/build/Release/"
        )
    endforeach()
endif()
