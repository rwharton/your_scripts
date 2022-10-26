import your as yr
import my_writer as my
import numpy as np
from argparse import ArgumentParser
import time


def truncate_fil(infile, outbase, flo, fhi):
    """
    Truncate (in frequency) the input filterbank 
    infile and write to outbase.fil

    Will output a filterbank starting with the 
    closest channel at or below flo and the 
    closest channel at or above fhi.
    """
    in_yr = yr.Your(infile) 

    fch1   = in_yr.fch1
    foff   = in_yr.foff
    nchans = in_yr.nchans

    freqs = np.arange(nchans) * foff + fch1

    # By default take the whole thing
    xx_min = 0
    xx_max = nchans

    if foff < 0:
        # Highest frequency is first
        xx_hi = np.where( freqs >= fhi )[0][-1]
        xx_lo = np.where( freqs <= flo )[0][0]

        xx_min = xx_hi
        xx_max = xx_lo
        
    else:
        # Highest frequency is last
        xx_hi = np.where( freqs >= fhi )[0][0]
        xx_lo = np.where( freqs <= flo )[0][-1]

        xx_min = xx_lo
        xx_max = xx_hi

    if xx_min == xx_max:
        print("No data selected!")
        return
    else: pass
    
    my_writer = my.Writer(in_yr, 
                          c_min = xx_min, 
                          c_max = xx_max, 
                          outdir = './', 
                          outname = outbase)

    my_writer.to_fil()
    return 


def parse_input():
    """
    Use argparse to parse input
    """
    prog_desc = "Truncate filterbank in frequency"
    parser = ArgumentParser(description=prog_desc)
    
    parser.add_argument('infile', help='Input *.fil file')
    parser.add_argument('-L', '--lofreq', 
                        help='Lowest frequency in output (MHz)', 
                        required=True, type=float)
    parser.add_argument('-H', '--hifreq', 
                        help='Highest frequency in output (MHz)',
                        required=True, type=float)
    parser.add_argument('-o', '--outbase', 
                        help='Output file basename (no suffix)', 
                        required=True)
     
    args = parser.parse_args()

    infile = args.infile
    lofreq = args.lofreq
    hifreq = args.hifreq
    outbase = args.outbase

    return infile, outbase, lofreq, hifreq


if __name__ == "__main__":
    tstart = time.time()
    infile, outbase, lofreq, hifreq = parse_input()
    truncate_fil(infile, outbase, lofreq, hifreq)
    tstop = time.time()

    dt = tstop - tstart
    print("Took %.2f minutes" %(dt/60.0))
