from collections import defaultdict
from argparse import ArgumentParser

def main():
    
    parser = ArgumentParser(description="")
    parser.add_argument("in_file", default="input.txt", help="reference file for evaluation")
    parser.add_argument("out_file", default="prediction.txt", help="prediction output file")
    
    args = parser.parse_args()

    input = []
    output = []

    with open(args.in_file, encoding="utf-8") as f:
        for line in f:
            # TODO
            raise

                    
    with open(args.out_file, encoding="utf-8") as f:
        for line in f:
            # TODO
            raise

    # TODO - compute eval
    raise
        
if __name__ == "__main__":
    main()

