#python.dataScience.textOutputLimit  = 0
from onnx_generator import args
args.verbose = 1
#args.header = ["include/operators"]
#args.no_header = 0
#args.check = ["src/operators/check"]
#args.no_check = 0
#args.resolve = ["src/operators/resolve"]
#args.no_resolve = 0
#args.sets = ["src/operators/"]
#args.no_sets = 0
#args.info_src = ["src/operators/info"]
#args.no_info_src = 0
#args.info_header = ["include/operators/info"]
#args.no_info_header = 0
#args.force = 0
#args.dryrun = 1
#args.include = [".*"]
#args.exclude = []
#args.version = ["all"]
args.domains  = ["ai.onnx"]
args.path = ["/home/mc/erp/data/cONNXr/"]

from onnx_generator import run
