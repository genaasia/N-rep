import argparse
import json
import os


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="inference data")
    parser.add_argument("--input-file", type=str, required=True, help="path to input json file")
    parser.add_argument("--lines-per", type=int, required=True, help="number of lines ")
    args = parser.parse_args()

    # validate
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"input file not found: {args.input_file}")

    with open(args.input_file, "r") as f:
        data = json.load(f)
    
    # split data
    for i in range(0, len(data), args.lines_per):
        chunk = data[i:i+args.lines_per]
        chunk_fname = args.input_file.replace(".json", f"_chunk-{i+1:08d}-{min(len(data), i+args.lines_per):08d}.json")
        with open(chunk_fname, "w") as f:
            json.dump(chunk, f, indent=2)
        print(f"saved {chunk_fname}")




