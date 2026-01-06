from src.preprocess.pose2d import Pose2dDetector

class Face2dDetector(Pose2dDetector):
    def __call__(self, *args, **kwargs):
        kwargs["mode"] = "face"
        return super().__call__(*args, **kwargs)
