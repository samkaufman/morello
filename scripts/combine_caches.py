#!python3
import argparse
import pickle
import sys
import tqdm
import os

from morello import search_cache

parser = argparse.ArgumentParser()
parser.add_argument("inputs", nargs="+", type=argparse.FileType("rb"))
parser.add_argument("output", type=argparse.FileType("wb"))


def main() -> int:
    parsed_args = parser.parse_args()

    sum_bytes_to_read = 0
    input_file_sizes = []
    for inp_fo in parsed_args.inputs:
        input_file_sizes.append(os.path.getsize(inp_fo.name))
        sum_bytes_to_read += input_file_sizes[-1]

    with tqdm.tqdm(
        total=sum_bytes_to_read, unit="B", unit_scale=True, unit_divisor=1024
    ) as progress_bar:
        accumulated_cache = search_cache.ScheduleCache()
        for inp_fo, inp_size in zip(parsed_args.inputs, input_file_sizes):
            progress_bar.write(f"Processing: {inp_fo.name}")
            loaded_cache = pickle.load(inp_fo)
            if not isinstance(loaded_cache, search_cache.ScheduleCache):
                progress_bar.write(
                    "Input file was not a ScheduleCache", file=sys.stderr
                )
                return 1
            accumulated_cache.update(loaded_cache)
            progress_bar.update(inp_size)

    print(f"Saving combined cache: {parsed_args.output.name}")
    pickle.dump(accumulated_cache, parsed_args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
