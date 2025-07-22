from datatrove.data import Document
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.writers import JsonlWriter


import ray
from datatrove.executor import RayPipelineExecutor
ray.init()
executor = RayPipelineExecutor(
    pipeline=[
        [
            Document(text="some data", id="0"),
            Document(text="some more data", id="1"),
            Document(text="even more data", id="2"),
        ],
        SamplerFilter(rate=0.5),
        JsonlWriter(
            output_folder="./artifacts/datatrove-test"
        )
    ],
    logging_dir="logs/",
    tasks=500,
    workers=100,  # omit to run all at once
)
executor.run()