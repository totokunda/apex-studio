{
  "targets": [{
    "target_name": "demux",
    "sources": [
      "csrc/demux.cpp",
    ],
    "include_dirs": [
      "<!@(node -p \"require('node-addon-api').include\")",
    ],
    "dependencies": [
      "<!(node -p \"require('node-addon-api').gyp\")"
    ],
    "defines": [ "NAPI_DISABLE_CPP_EXCEPTIONS" ],
    "cflags!": [ "-fno-exceptions" ],
    "cflags_cc!": [ "-fno-exceptions" ],
    "conditions": [
       ["OS=='mac'", {
        "include_dirs": [
          "<!@(pkg-config --cflags-only-I libavcodec libavformat libavutil libswscale | sed 's/-I//g')"
        ],
        "libraries": [
          "<!@(pkg-config --libs libavcodec libavformat libavutil libswscale)",
          "-framework VideoToolbox",
          "-framework CoreMedia",
          "-framework CoreVideo",
          "-framework CoreFoundation"
        ],
        "xcode_settings": {
          "OTHER_CPLUSPLUSFLAGS": ["-std=c++17", "-march=armv8.2-a+dotprod+i8mm"],
          "OTHER_CFLAGS": ["-march=armv8.2-a+dotprod+i8mm"],
          "GCC_ENABLE_CPP_EXCEPTIONS": "YES"
        }
      }],
      ["OS=='linux'", {
        "libraries": [
          "-lavcodec",
          "-lavformat",
          "-lavutil",
          "-lswscale"
        ],
        "cflags_cc": ["-std=c++17"]
      }],
      ["OS=='win'", {
        "libraries": [
          "avcodec.lib",
          "avformat.lib",
          "avutil.lib",
          "swscale.lib"
        ],
        "include_dirs": [
          "C:/ffmpeg/include"
        ],
        "library_dirs": [
          "C:/ffmpeg/lib"
        ]
      }]
    ]
  }]
}
