"""Use the frozen R164 monitor/export semantics with the new paired provenance."""
from pathlib import Path
import importlib.util,sys
HERE=Path(__file__).resolve().parent
def main():
    source=HERE/('export_recovered.py' if sys.argv[1]=='export' else 'monitor.py')
    spec=importlib.util.spec_from_file_location('r169_bound_workflow',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if sys.argv[1]=='export':module.main()
    else:getattr(module,sys.argv[1])(sys.argv[2])
if __name__=='__main__':main()
