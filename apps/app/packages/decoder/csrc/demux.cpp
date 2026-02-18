#include <napi.h>
#include <cstdio>

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/avutil.h>
    #include <libavutil/rational.h>
}

// Build WebCodecs codec string from AVCodecParameters
static std::string codecString(const AVCodecParameters* cp) {
    if (!cp) return "";

    switch (cp->codec_id) {
        case AV_CODEC_ID_H264: {
            // avc1.PPCCLL from avcC: profile, profile_compatibility, level
            if (cp->extradata_size >= 4) {
                char buf[32];
                snprintf(buf, sizeof(buf), "avc1.%02X%02X%02X",
                    (unsigned)cp->extradata[1],
                    (unsigned)cp->extradata[2],
                    (unsigned)cp->extradata[3]);
                return buf;
            }
            return "avc1.42E01E";  // Fallback: constrained baseline level 3
        }
        case AV_CODEC_ID_HEVC:
            return "hev1.1.6.L93.B0";  // Common fallback
        case AV_CODEC_ID_VP8:
            return "vp08";
        case AV_CODEC_ID_VP9:
            return "vp09";
        case AV_CODEC_ID_AV1:
            return "av01.0.04M.08";
        case AV_CODEC_ID_AAC:
            return "mp4a";
        case AV_CODEC_ID_OPUS:
            return "opus";
        default:
            return "";
    }
}

// Convert pts in stream time_base to microseconds
static int64_t ptsToMicros(int64_t pts, const AVRational& timeBase) {
    if (pts == AV_NOPTS_VALUE) return -1;
    return av_rescale_q(pts, timeBase, AVRational{1, 1000000});
}

Napi::Value LoadFile(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Filename (string) required").ThrowAsJavaScriptException();
        return env.Null();
    }

    std::string filename = info[0].As<Napi::String>().Utf8Value();

    AVFormatContext* fmtCtx = nullptr;
    if (avformat_open_input(&fmtCtx, filename.c_str(), nullptr, nullptr) < 0) {
        Napi::Error::New(env, "Failed to open file").ThrowAsJavaScriptException();
        return env.Null();
    }

    if (avformat_find_stream_info(fmtCtx, nullptr) < 0) {
        avformat_close_input(&fmtCtx);
        Napi::Error::New(env, "Failed to read stream info").ThrowAsJavaScriptException();
        return env.Null();
    }

    // Build streams array with WebCodecs metadata
    Napi::Array streams = Napi::Array::New(env);

    for (unsigned int i = 0; i < fmtCtx->nb_streams; i++) {
        AVStream* stream = fmtCtx->streams[i];
        AVCodecParameters* cp = stream->codecpar;

        Napi::Object s = Napi::Object::New(env);
        s.Set("index", Napi::Number::New(env, i));
        s.Set("codecType", Napi::String::New(env, cp->codec_type == AVMEDIA_TYPE_VIDEO ? "video" :
                                              cp->codec_type == AVMEDIA_TYPE_AUDIO ? "audio" : "unknown"));
        s.Set("timeBaseNum", Napi::Number::New(env, stream->time_base.num));
        s.Set("timeBaseDen", Napi::Number::New(env, stream->time_base.den));
        s.Set("duration", Napi::Number::New(env, stream->duration != AV_NOPTS_VALUE ? static_cast<double>(stream->duration) : -1));
        s.Set("durationMicros", Napi::Number::New(env, stream->duration != AV_NOPTS_VALUE ? ptsToMicros(stream->duration, stream->time_base) : -1));

        s.Set("codec", Napi::String::New(env, codecString(cp)));

        if (cp->codec_type == AVMEDIA_TYPE_VIDEO) {
            s.Set("codedWidth", Napi::Number::New(env, cp->width));
            s.Set("codedHeight", Napi::Number::New(env, cp->height));
            s.Set("videoDecoderConfig", Napi::Object::New(env));
            Napi::Object vdc = s.Get("videoDecoderConfig").As<Napi::Object>();
            vdc.Set("codec", Napi::String::New(env, codecString(cp)));
            vdc.Set("codedWidth", Napi::Number::New(env, cp->width));
            vdc.Set("codedHeight", Napi::Number::New(env, cp->height));
            if (cp->extradata_size > 0 && cp->extradata) {
                vdc.Set("description", Napi::Buffer<uint8_t>::Copy(env, cp->extradata, cp->extradata_size));
            }
        } else if (cp->codec_type == AVMEDIA_TYPE_AUDIO) {
            s.Set("sampleRate", Napi::Number::New(env, cp->sample_rate));
            s.Set("channelCount", Napi::Number::New(env, cp->ch_layout.nb_channels));
            if (cp->extradata_size > 0 && cp->extradata) {
                s.Set("description", Napi::Buffer<uint8_t>::Copy(env, cp->extradata, cp->extradata_size));
            }
        }

        streams[i] = s;
    }

    // Build packets array (WebCodecs-ready)
    Napi::Array packets = Napi::Array::New(env);
    uint32_t count = 0;

    AVPacket* pkt = av_packet_alloc();
    if (!pkt) {
        avformat_close_input(&fmtCtx);
        Napi::Error::New(env, "Failed to allocate packet").ThrowAsJavaScriptException();
        return env.Null();
    }

    while (av_read_frame(fmtCtx, pkt) >= 0) {
        AVStream* stream = fmtCtx->streams[pkt->stream_index];

        Napi::Object obj = Napi::Object::New(env);
        obj.Set("streamIndex", Napi::Number::New(env, pkt->stream_index));
        obj.Set("isKeyFrame", Napi::Boolean::New(env, (pkt->flags & AV_PKT_FLAG_KEY) != 0));
        obj.Set("timestampMicros", Napi::Number::New(env, ptsToMicros(pkt->pts, stream->time_base)));
        obj.Set("pts", Napi::Number::New(env, pkt->pts == AV_NOPTS_VALUE ? -1 : static_cast<double>(pkt->pts)));
        obj.Set("dts", Napi::Number::New(env, pkt->dts == AV_NOPTS_VALUE ? -1 : static_cast<double>(pkt->dts)));

        if (pkt->size > 0 && pkt->data) {
            obj.Set("data", Napi::Buffer<uint8_t>::Copy(env, pkt->data, pkt->size));
        } else {
            obj.Set("data", Napi::Buffer<uint8_t>::New(env, 0));
        }

        packets[count++] = obj;
        av_packet_unref(pkt);
    }

    av_packet_free(&pkt);

    // Container duration in seconds
    int64_t durationUs = fmtCtx->duration != AV_NOPTS_VALUE ? fmtCtx->duration : -1;
    double durationSec = durationUs >= 0 ? durationUs / 1000000.0 : -1;

    avformat_close_input(&fmtCtx);

    Napi::Object result = Napi::Object::New(env);
    result.Set("streams", streams);
    result.Set("packets", packets);
    result.Set("duration", Napi::Number::New(env, durationSec));
    result.Set("durationMicros", Napi::Number::New(env, static_cast<double>(durationUs)));

    return result;
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set(Napi::String::New(env, "loadFile"), Napi::Function::New(env, LoadFile));
    return exports;
}

NODE_API_MODULE(demux, Init)
