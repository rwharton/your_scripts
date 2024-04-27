import your as yr
from your.formats.filwriter import make_sigproc_object
import numpy as np
from argparse import ArgumentParser
import time
import sys
from astropy import units as u
from astropy.coordinates import SkyCoord

def coord_to_raj_dej(cc):
    """
    From SkyCoord object, get the "raj" and "dej" 
    floats used in SIGPROC:

         08h35m20.611s -> 83520.611
        -45d10m34.87s  -> -451034.87
    """
    rh, rm, rs = cc.ra.hms
    raj = rh * 1e4 + rm * 1e2 + rs

    dd, dm, ds = cc.dec.dms
    dej = dd * 1e4 + dm * 1e2 + ds
    return raj, dej


def fix_hdr(args):
    """
    Fix select header values by writing new file

    Can fix:
        * telescope_id
        * machine_id 
        * source_name
    """
    in_yr = yr.Your(args.infile) 

    outfile = "%s.fil" %args.outbase

    if args.telescope_id is not None:
        in_yr.telescope_id = args.telescope_id
    
    if args.machine_id is not None:
        in_yr.machine_id = args.machine_id
    
    if args.source_name is not None:
        in_yr.your_header.source_name = args.source_name

    if args.pulsarcentric is not None:
        in_yr.pulsarcentric = args.pulsarcentric

    if (args.ra_str is not None) and (args.dec_str is not None):
        ra_str = args.ra_str
        dec_str = args.dec_str 

        c = SkyCoord(ra_str, dec_str, frame='icrs', unit=(u.hourangle, u.deg))

        raj, dej = coord_to_raj_dej(c)
        
        in_yr.src_raj = raj
        in_yr.src_dej = dej
    
        ra_deg = c.ra.deg
        dec_deg = c.dec.deg
    
        in_yr.ra_deg = ra_deg
        in_yr.dec_deg = dec_deg

        in_yr.your_header.ra_deg = ra_deg
        in_yr.your_header.dec_deg = dec_deg
        
        print(in_yr.your_header)

    hdr_in = in_yr.your_header 

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
            nbits         = in_yr.nbits,
            nifs          = in_yr.nifs,
            barycentric   = in_yr.barycentric,
            pulsarcentric = in_yr.pulsarcentric,
            telescope_id  = in_yr.telescope_id,
            data_type     = in_yr.data_type,
            az_start      = in_yr.az_start,
            za_start      = in_yr.za_start)

    # Write header
    sig_obj.write_header(outfile)

    nbits  = in_yr.nbits
    nchans = hdr_in.nchans
    nsamp  = hdr_in.native_nspectra 

    max_chunk_gb = 5.0
    max_chunk_bits = max_chunk_gb * 8 * 1e9
    chunk_nsamp  = int( max_chunk_bits / (nchans * nbits))
    if chunk_nsamp < 1:
        chunk_nsamp = 1
    
    print("nsamples    = %d" %nsamp)
    print("chunk_nsamp = %d" %chunk_nsamp)

    nread = 0
    nsteps = int( nsamp / chunk_nsamp)
    if nsteps * chunk_nsamp < nsamp:
        nsteps += 1
    else: pass

    for ii in range(nsteps):
        if nsamp - nread < chunk_nsamp:
            nii = nsamp - nread
        else:
            nii = chunk_nsamp
        print("nread = %d" %nread)
        print("nii   = %d" %nii)

        # write data
        sig_obj.append_spectra(in_yr.get_data(nread, nii), outfile)

        nread += nii

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
    parser.add_argument('-pc', '--pulsarcentric',
                        help='Pulsarcentric (0=no, 1=yes)',
                        required=False, type=int)
    parser.add_argument('-ra', '--ra_str', 
                        help='RA string with format hh:mm:ss',
                        required=False)
    parser.add_argument('-dec', '--dec_str', 
                        help='Dec string with format +dd:mm:ss',
                        required=False)
    parser.add_argument('-o', '--outbase', 
                        help='Output file basename (no suffix)', 
                        required=True)
     
    args = parser.parse_args()

    return args

debug = 0

if __name__ == "__main__":
    if debug:
        sys.exit(0)
    else: pass

    tstart = time.time()
    args = parse_input()
    fix_hdr(args)
    tstop = time.time()

    dt = tstop - tstart
    print("Took %.2f minutes" %(dt/60.0))
