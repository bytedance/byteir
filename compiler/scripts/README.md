## Generate testcases

### E2E testcases
- python3 scripts/gen_testcases.py --top-dir=test/E2E --category=E2E

### Host pipeline testcases
- python3 scripts/gen_testcases.py --top-dir=test/Pipelines/Host/E2E --category=HostPipeline


### Add new template-based testcase
- create template.py as input testcase template file
- define `Testcase` with `contents` which was a list of `Content` and `pipelines` which was a list of `Pipeline`
- each testcase content should be a `.mlir` content attaching to several `Stage`s (e.g. Input stage)
- each pipeline describe a pass pipeline from one stage to other stages with `filecheck` string
- there was some predefined `Stage`s and `Pipeline`s in `gen_testcases.py` and were categoried into different collections(e.g. E2ECollections and HostPipelineCollections), but it's fine to define new `Stage` or `Pipeline` in template file