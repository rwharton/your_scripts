import your as yr
import my_writer as my
import numpy as np
from argparse import ArgumentParser
import time


def fix_hdr(args):
    """
    Fix select header values by writing new file

    Can fix:
        * telescope_id
        * machine_id 
        * source_name
    """
    in_yr = yr.Your(args.infile) 

    if args.telescope_id is not None:
        in_yr.telescope_id = args.telescope_id
    
    if args.machine_id is not None:
        in_yr.machine_id = args.machine_id
    
    if args.source_name is not None:
        in_yr.your_header.source_name = args.source_name

    my_writer = my.Writer(in_yr, 
                          outdir = './', 
                          outname = args.outbase)

    my_writer.to_fil()
    return 


def parse_input():
    """
    Use argparse to parse input
    """
    prog_desc = "Truncate filterbank in frequency"
    parser = ArgumentParser(description=prog_desc)
    
    parser.add_argument('infile', help='Input *.fil file')
    parser.add_argument('-tel', '--telescope_id', 
                        help='Telescope ID Number', 
                        required=False, type=int)
    parser.add_argument('-m', '--machine_id', 
                        help='Machine ID Number',
                        required=False, type=int)
    parser.add_argument('-src', '--source_name', 
                        help='Source Name',
                        required=False)
    parser.add_argument('-o', '--outbase', 
                        help='Output file basename (no suffix)', 
                        required=True)
     
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    tstart = time.time()
    args = parse_input()
    fix_hdr(args)
    tstop = time.time()

    dt = tstop - tstart
    print("Took %.2f minutes" %(dt/60.0))
