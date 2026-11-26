"""Run the same Nano detector with the opt-in R100 PF estimate."""
import sys

from bench_uniform_pf import estimate_uniform, lse_r98, pf_ctrv, register_pf

register_pf()
pf_ctrv._logsumexp = lse_r98
pf_ctrv.PFCTRV.estimate_state = estimate_uniform

import nano_live  # noqa: E402

if __name__ == "__main__":
    sys.exit(nano_live.main())
