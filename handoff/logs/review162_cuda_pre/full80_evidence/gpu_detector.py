"""Only the detector adapter changes; decoding/NMS/scaling remain imported."""
import time
from nano_detector import NanoDetector,decode_nms,scale_boxes

class GpuNanoDetector(NanoDetector):
    def detect(self,img_bgr):
        assert self.roi_y is None,'CUDA trial does not use an ROI crop'
        started=time.perf_counter()
        raw=self.infer(img_bgr)
        inferred=time.perf_counter()
        det=decode_nms(raw,self.conf,self.iou,self.max_det,self.classes,self.agnostic)
        det=scale_boxes(det,self.infer.hw,img_bgr.shape[:2])
        finished=time.perf_counter()
        # GPU preprocessing and raw upload are inside infer_ms. They must not
        # be presented as eliminated simply because CPU pre_ms is zero.
        measured={'pre_ms':0.,'infer_ms':1000*(inferred-started),'post_ms':1000*(finished-inferred)}
        for key in measured:self.t[key]+=measured[key]
        self.t['n']+=1
        return det,measured
