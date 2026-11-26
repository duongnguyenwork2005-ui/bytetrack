"""Forward the current score without changing shared Kalman instance identity."""
from contextlib import contextmanager
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
import ul_backend
import tracker_ctrv as tc


@contextmanager
def score_scope(kf,score,forward=True,audit=None):
    score=float(score)
    if not np.isfinite(score) or not 0<=score<=1:raise ValueError('Current detection score in [0,1] required')
    owned='update' in kf.__dict__;local=kf.__dict__.get('update');original=kf.update
    def update(mean,covariance,measurement,confidence=None):
        delivered=score if confidence is None and forward else confidence
        if audit is not None:audit(score,delivered)
        return original(mean,covariance,measurement,confidence=delivered)
    kf.update=update
    try:yield
    finally:
        if owned:kf.update=local
        else:del kf.update


def track_class(base,forward=True,audit=None):
    class ScoreTrack(base):
        def update(self,new_track,frame_id):
            with score_scope(self.kalman_filter,new_track.score,forward,audit):
                return super().update(new_track,frame_id)
        def re_activate(self,new_track,frame_id,new_id=False):
            with score_scope(self.kalman_filter,new_track.score,forward,audit):
                return super().re_activate(new_track,frame_id,new_id)
    return ScoreTrack


@contextmanager
def installed(forward=True,audit=None):
    trackers=(ul_backend.BYTETracker,tc.CTRVByteTracker,tc.UKFByteTracker)
    originals=[(cls,cls.__dict__.get('track_class'),cls.track_class) for cls in trackers]
    for cls,_,base in originals:cls.track_class=track_class(base,forward,audit)
    try:yield
    finally:
        for cls,owned,base in originals:
            if owned is None:delattr(cls,'track_class')
            else:cls.track_class=owned
