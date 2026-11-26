"""Copy the unused R128 confirmation harness into R132 with explicit model names."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
source=(HERE.parent/'review128_pf_heading/test40_job.py').read_text()
assert "from heading import installed" in source and source.count('heading100')==5
assert "write(OUT/'inputs.json',{'frames':frames,'configs':configs,'sha256'" in source
source=source.replace('R128','R132').replace('from heading import installed','from predicted_noise import installed').replace('heading100','candidate100')
source=source.replace("HERE/'01_test40_confirmation.md',MANIFEST", "HERE/'01_test40_confirmation.md',HERE/'test40_bootstrap.py',MANIFEST")
source=source.replace("{'frames':frames,'configs':configs,'sha256'", "{'frames':frames,'configs':configs,'effective_measurement_noise_height':'prediction','sha256'")
target=HERE/'test40_job.py'
assert not target.exists()
target.write_text(source)
print('Prepared R132 confirmation harness',len(source),flush=True)
