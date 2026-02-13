#include "apple_vt_reader.h"

#if defined(__APPLE__)

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>

#include <cmath>
#include <limits>

namespace apex {

static double ToSeconds(CMTime time, double fallback = 0.0) {
    if (!CMTIME_IS_VALID(time) || CMTIME_IS_INDEFINITE(time)) return fallback;
    const double seconds = CMTimeGetSeconds(time);
    if (!std::isfinite(seconds)) return fallback;
    return seconds;
}

bool AppleVTDecodeOnlyIterate(
    const std::string& filePath,
    double startTime,
    double endTime,
    std::atomic<bool>& cancelled,
    const std::function<bool(double timestamp, double duration)>& onFrame,
    std::string& error
) {
    @autoreleasepool {
        NSString* path = [NSString stringWithUTF8String:filePath.c_str()];
        if (!path || [path length] == 0) {
            error = "Invalid file path for Apple VideoToolbox reader";
            return false;
        }

        NSURL* url = [NSURL fileURLWithPath:path];
        AVURLAsset* asset = [AVURLAsset URLAssetWithURL:url options:nil];
        NSArray<AVAssetTrack*>* videoTracks = [asset tracksWithMediaType:AVMediaTypeVideo];
        if (!videoTracks || [videoTracks count] == 0) {
            error = "No video track found for Apple VideoToolbox reader";
            return false;
        }

        AVAssetTrack* videoTrack = [videoTracks firstObject];
        NSError* readerErr = nil;
        AVAssetReader* reader = [[AVAssetReader alloc] initWithAsset:asset error:&readerErr];
        if (!reader) {
            error = readerErr ? [[readerErr localizedDescription] UTF8String] : "Failed to create AVAssetReader";
            return false;
        }

        NSDictionary* outputSettings = @{
            (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange),
        };
        AVAssetReaderTrackOutput* output =
            [[AVAssetReaderTrackOutput alloc] initWithTrack:videoTrack outputSettings:outputSettings];
        output.alwaysCopiesSampleData = NO;

        if (![reader canAddOutput:output]) {
            error = "AVAssetReader cannot add video track output";
            return false;
        }
        [reader addOutput:output];

        const double safeStart = std::max(0.0, startTime);
        CMTime start = CMTimeMakeWithSeconds(safeStart, 600);
        CMTimeRange range = kCMTimeRangeInvalid;
        if (std::isfinite(endTime) && endTime > safeStart) {
            CMTime end = CMTimeMakeWithSeconds(endTime, 600);
            range = CMTimeRangeFromTimeToTime(start, end);
        } else {
            range = CMTimeRangeMake(start, kCMTimePositiveInfinity);
        }
        if (CMTIMERANGE_IS_VALID(range)) {
            reader.timeRange = range;
        }

        if (![reader startReading]) {
            NSError* startErr = [reader error];
            error = startErr ? [[startErr localizedDescription] UTF8String] : "AVAssetReader failed to start";
            return false;
        }

        while (!cancelled.load(std::memory_order_relaxed)) {
            CMSampleBufferRef sample = [output copyNextSampleBuffer];
            if (!sample) break;

            const double timestamp = ToSeconds(
                CMSampleBufferGetPresentationTimeStamp(sample),
                0.0
            );
            const double duration = ToSeconds(
                CMSampleBufferGetDuration(sample),
                0.0
            );

            const bool keepGoing = onFrame(timestamp, duration);
            CFRelease(sample);

            if (!keepGoing) {
                [reader cancelReading];
                return true;
            }
        }

        if (cancelled.load(std::memory_order_relaxed)) {
            [reader cancelReading];
            return true;
        }

        AVAssetReaderStatus status = [reader status];
        if (status == AVAssetReaderStatusFailed) {
            NSError* readErr = [reader error];
            error = readErr ? [[readErr localizedDescription] UTF8String] : "AVAssetReader failed";
            return false;
        }

        return true;
    }
}

} // namespace apex

#else

namespace apex {

bool AppleVTDecodeOnlyIterate(
    const std::string&,
    double,
    double,
    std::atomic<bool>&,
    const std::function<bool(double, double)>&,
    std::string& error
) {
    error = "Apple VideoToolbox reader is only available on macOS";
    return false;
}

} // namespace apex

#endif
