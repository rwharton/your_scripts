import your as yr
from your.formats.filwriter import make_sigproc_object
import numpy as np
from argparse import ArgumentParser
import time


def fix_file(args):
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

    hdr_in = in_yr.your_header

    outfile = "%s.fil" %args.outbase

    # Setup output filterbank file
    sig_obj = make_sigproc_object(
            rawdatafile   = outfile,
            source_name   = hdr_in.source_name,
            nchans        = hdr_in.nchans, 
            foff          = hdr_in.foff, 
            fch1          = hdr_in.fch1, 
            tsamp         = hdr_in.tsamp, 
            tstart        = hdr_in.tstart, 
            src_raj       = in_yr.src_raj, 
            src_dej       = in_yr.src_dej, 
            machine_id    = in_yr.machine_id,
            nbeams        = in_yr.nbeams,
            ibeam         = in_yr.ibeam,
            nbits         = 8,
            nifs          = in_yr.nifs,
            barycentric   = in_yr.barycentric,
            pulsarcentric = in_yr.pulsarcentric ,
            telescope_id  = in_yr.telescope_id,
            data_type     = in_yr.data_type,
            az_start      = in_yr.az_start,
            za_start      = in_yr.za_start)

    # Write header
    sig_obj.write_header(outfile)

    # Loop over input, rescale, convert to 8bit, write
    nspec_total = hdr_in.nspectra
    nspec_chunk = 10**5
    nsteps = nspec_total // nspec_chunk
    if nspec_total > (nspec_chunk * nsteps):
        nsteps += 1 
    else: pass

    for ii in range(nsteps):
        # get input data
        start_ii = ii * nspec_chunk 
        indat = in_yr.get_data(nstart=start_ii, nsamp=nspec_chunk) 
    
        # scale + offset
        outdat = (indat * (256/8) + 128).astype('uint8')

        # write data 
        sig_obj.append_spectra(outdat, outfile)

    return 


def parse_input():
    """
    Use argparse to parse input
    """
    prog_desc = "Convert to 8-bit (can also change some header vals if you want)"
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
    fix_file(args)
    tstop = time.time()

    dt = tstop - tstart
    print("Took %.2f minutes" %(dt/60.0))
