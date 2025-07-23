import os
import ray
import argparse

from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
from datatrove.executor import RayPipelineExecutor

parser = argparse.ArgumentParser(description="Ray Test Pipeline")
parser.add_argument("--artifact_path", type=str,
                    default="./artifacts/datatrove-test",
                    help="Path to store artifacts")
parser.add_argument("--ray_temp_dir", type=str,
                    default="/data/temp/ray",
                    help="Temporary directory for Ray, recommended to use a path on the same storage as artifact_path")
args = parser.parse_args()

if args.ray_temp_dir == "/data/temp/ray":
    print("\033[93mWARNING:\033[0m Using default ray_temp_dir. It is recommended to set this to a path on the same storage as artifact_path for better performance.")

logging_dir = os.path.abspath(os.path.join(args.artifact_path, "logs"))
output_dir = os.path.abspath(os.path.join(args.artifact_path, "data"))
ray_temp_dir = os.path.abspath(args.ray_temp_dir)

ray.init(_temp_dir=ray_temp_dir)

executor = RayPipelineExecutor(
    pipeline=[
        ParquetReader(
            "hf://datasets/HuggingFaceFW/fineweb-2/data/kor_Hang/train",
            limit=20000,
        ),
        SamplerFilter(rate=0.1),
        ParquetWriter(
            output_folder=output_dir
        )
    ],
    logging_dir=logging_dir,
    tasks=500,
    workers=100,
)
executor.run()