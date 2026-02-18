cmd_Release/obj.target/addon/src/addon.o := c++ -o Release/obj.target/addon/src/addon.o ../src/addon.cpp '-DNODE_GYP_MODULE_NAME=addon' '-DUSING_UV_SHARED=1' '-DUSING_V8_SHARED=1' '-DV8_DEPRECATION_WARNINGS=1' '-D_GLIBCXX_USE_CXX11_ABI=1' '-D_FILE_OFFSET_BITS=64' '-D_DARWIN_USE_64_BIT_INODE=1' '-D_LARGEFILE_SOURCE' '-DNAPI_DISABLE_CPP_EXCEPTIONS' '-DBUILDING_NODE_EXTENSION' -I/Users/tosinkuye/Library/Caches/node-gyp/25.6.1/include/node -I/Users/tosinkuye/Library/Caches/node-gyp/25.6.1/src -I/Users/tosinkuye/Library/Caches/node-gyp/25.6.1/deps/openssl/config -I/Users/tosinkuye/Library/Caches/node-gyp/25.6.1/deps/openssl/openssl/include -I/Users/tosinkuye/Library/Caches/node-gyp/25.6.1/deps/uv/include -I/Users/tosinkuye/Library/Caches/node-gyp/25.6.1/deps/zlib -I/Users/tosinkuye/Library/Caches/node-gyp/25.6.1/deps/v8/include -I/Users/tosinkuye/apex-workspace/apex-studio/apps/app/node_modules/node-addon-api -I/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7  -O3 -gdwarf-2 -fno-strict-aliasing -mmacosx-version-min=13.5 -arch arm64 -Wall -Wendif-labels -W -Wno-unused-parameter -std=gnu++20 -stdlib=libc++ -fno-rtti -std=c++17 -MMD -MF ./Release/.deps/Release/obj.target/addon/src/addon.o.d.raw -I/opt/homebrew/opt/ffmpeg-full/include  -c
Release/obj.target/addon/src/addon.o: ../src/addon.cpp \
  /Users/tosinkuye/apex-workspace/apex-studio/apps/app/node_modules/node-addon-api/napi.h \
  /Users/tosinkuye/Library/Caches/node-gyp/25.6.1/include/node/node_api.h \
  /Users/tosinkuye/Library/Caches/node-gyp/25.6.1/include/node/js_native_api.h \
  /Users/tosinkuye/Library/Caches/node-gyp/25.6.1/include/node/js_native_api_types.h \
  /Users/tosinkuye/Library/Caches/node-gyp/25.6.1/include/node/node_api_types.h \
  /Users/tosinkuye/apex-workspace/apex-studio/apps/app/node_modules/node-addon-api/napi-inl.h \
  /Users/tosinkuye/apex-workspace/apex-studio/apps/app/node_modules/node-addon-api/napi-inl.deprecated.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_animation.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_export.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_property.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_types.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_pool.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_audio.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_cache.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_chain.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_link.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_producer.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_filter.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_service.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_properties.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_events.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_profile.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_consumer.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_deque.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_factory.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_repository.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_field.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_frame.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_image.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_log.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_multitrack.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_parser.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_playlist.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_slices.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_tokeniser.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_tractor.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_transition.h \
  /opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_version.h
../src/addon.cpp:
/Users/tosinkuye/apex-workspace/apex-studio/apps/app/node_modules/node-addon-api/napi.h:
/Users/tosinkuye/Library/Caches/node-gyp/25.6.1/include/node/node_api.h:
/Users/tosinkuye/Library/Caches/node-gyp/25.6.1/include/node/js_native_api.h:
/Users/tosinkuye/Library/Caches/node-gyp/25.6.1/include/node/js_native_api_types.h:
/Users/tosinkuye/Library/Caches/node-gyp/25.6.1/include/node/node_api_types.h:
/Users/tosinkuye/apex-workspace/apex-studio/apps/app/node_modules/node-addon-api/napi-inl.h:
/Users/tosinkuye/apex-workspace/apex-studio/apps/app/node_modules/node-addon-api/napi-inl.deprecated.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_animation.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_export.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_property.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_types.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_pool.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_audio.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_cache.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_chain.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_link.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_producer.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_filter.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_service.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_properties.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_events.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_profile.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_consumer.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_deque.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_factory.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_repository.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_field.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_frame.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_image.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_log.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_multitrack.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_parser.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_playlist.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_slices.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_tokeniser.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_tractor.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_transition.h:
/opt/homebrew/Cellar/mlt/7.36.1_1/include/mlt-7/framework/mlt_version.h:
