{
  "targets": [
    {
      "target_name": "addon",
      "sources": ["src/addon.cpp"],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "defines": ["NAPI_DISABLE_CPP_EXCEPTIONS"],
      "cflags!": ["-fno-exceptions"],
      "cflags_cc!": ["-fno-exceptions"],
      "conditions": [
        [
          "OS=='mac'",
          {
            "include_dirs": [
              "<!@(pkg-config --cflags-only-I mlt-framework-7 | sed 's/-I//g')"
            ],
            "libraries": [
              "<!@(pkg-config --libs mlt-framework-7)"
            ],
            "xcode_settings": {
              "OTHER_CPLUSPLUSFLAGS": ["-std=c++17"],
              "GCC_ENABLE_CPP_EXCEPTIONS": "YES"
            }
          }
        ],
        [
          "OS=='linux'",
          {
            "include_dirs": [
              "<!@(pkg-config --cflags-only-I mlt-framework-7 | sed 's/-I//g')"
            ],
            "libraries": [
              "<!@(pkg-config --libs mlt-framework-7)"
            ],
            "cflags_cc": ["-std=c++17"]
          }
        ],
        [
          "OS=='win'",
          {
            "libraries": ["mlt-7.lib"],
            "include_dirs": ["C:/mlt/include"],
            "library_dirs": ["C:/mlt/lib"],
            "msvs_settings": {
              "VCCLCompilerTool": {
                "AdditionalOptions": ["/std:c++17"]
              }
            }
          }
        ]
      ]
    }
  ]
}
