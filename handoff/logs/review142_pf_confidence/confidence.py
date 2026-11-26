"""Forward the current matched detection's score to the existing PF R model."""
from contextlib import contextmanager
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'review132_pf_predicted_noise'))
from predicted_noise import PredictedNoisePF
import rbpf_ctrv as rb

PENDING='_r142_current_confidence'
MISSING=object()


@contextmanager
def score_scope(track,score):
    value=float(score)
    if not np.isfinite(value) or not 0<=value<=1:raise ValueError('Detection confidence in [0,1] required')
    previous=track.__dict__.get(PENDING,MISSING)
    setattr(track,PENDING,value)
    try:yield
    finally:
        if previous is MISSING:delattr(track,PENDING)
        else:setattr(track,PENDING,previous)


def view_class(forward=True,audit=None):
    class ConfidenceView(rb.RBPFTrackView):
        def update(self,mean,covariance,measurement,confidence=None):
            pending=getattr(self.track,PENDING,None)
            delivered=pending if forward and confidence is None else confidence
            if audit is not None:audit(pending,delivered)
            return super().update(mean,covariance,measurement,delivered)
    return ConfidenceView


def track_class(view):
    class ConfidenceTrack(rb.RBPFSTrack):
        def activate(self,pf,frame_id):
            super().activate(pf,frame_id)
            self.kalman_filter=view(pf,self)

        def update(self,new_track,frame_id):
            with score_scope(self,new_track.score):
                return super().update(new_track,frame_id)

        def re_activate(self,new_track,frame_id,new_id=False):
            with score_scope(self,new_track.score):
                return super().re_activate(new_track,frame_id,new_id)
    return ConfidenceTrack


@contextmanager
def installed(enabled=True,forward=True,audit=None):
    previous=rb.ConditionalPF;previous_track=rb.RBPFByteTracker.track_class
    rb.ConditionalPF=PredictedNoisePF
    rb.RBPFByteTracker.track_class=track_class(view_class(forward,audit)) if enabled else rb.RBPFSTrack
    try:yield
    finally:
        rb.ConditionalPF=previous
        rb.RBPFByteTracker.track_class=previous_track
