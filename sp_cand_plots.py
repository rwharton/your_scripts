import your 
from your.candidate import Candidate, crop
from your.utils.plotter import plot_h5
import numpy as np
from scipy.signal import detrend
import os
from argparse import ArgumentParser
from subprocess import call

import matplotlib.pyplot as plt 

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

##############################
## Making and Reading Cands ##
##############################

class SP_CAND:
    def __init__(self, line, ctype):
        self.dm = None
        self.snr = None
        self.time = None
        self.samp = None
        self.wbins = None

        if ctype == 'sp':
            self.input_sp(line)

        elif ctype == 'cand':
            self.input_cand(line)
    
        elif ctype == 'inj':
            self.input_inj(line)
        
        else:
            print("Unrecognized data type: %s" %ctype)

    def input_sp(self, line):
        cols = line.split()
        self.dm = float(cols[0])
        self.snr = float(cols[1])
        self.time = float(cols[2])
        self.samp = int(cols[3])
        self.wbins = int(cols[4])

    def input_cand(self, line):
        cols = line.split()
        self.dm = float(cols[6])
        self.snr = float(cols[0])
        self.time = float(cols[3])
        self.samp = int(cols[1])
        self.wbins = int(cols[4])

    def input_inj(self, line):
        cols = line.split()
        dm = float(cols[0])
        sigma = float(cols[1])
        time = float(cols[2])
        samp = int(cols[3])
        dfac = int(cols[4])
        dt   = float(cols[9]) * 1e-6
        fhi  = float(cols[11])
        fref = float(cols[13])
        
        tfix = 4.15e3 * (fref**-2.0 - fhi**-2.0) * dm
        nfix = int( tfix / dt )

        self.dm = dm
        self.snr = sigma
        self.time = time - tfix
        self.samp = samp - nfix 
        self.wbins = dfac
        

    def __str__(self):
        str = "SP_CAND(t=%.2f s, DM=%.1f pc/cc, SNR=%.2f)" %(self.time, self.dm, self.snr)
        return str

    def __repr__(self):
        str = "SP_CAND(t=%.2f s, DM=%.1f pc/cc, SNR=%.2f)" %(self.time, self.dm, self.snr)
        return str


def cands_from_spfile(spfile, ctype='sp'):
    """
    Read in the candidates from a *singlepulse (sp) file 
    or *cand (cand) file and return an array of SP_CAND 
    class objects
    """    
    candlist = []
    with open(spfile, 'r') as fin:
        for line in fin:
            if line[0] in ["\n", "#", 'i']:
                continue
            else: pass
    
            SP = SP_CAND(line, ctype)
            candlist.append(SP)

    return np.array(candlist)


#########################
##  Remove Duplicates  ##
#########################

def t_dm_shifts(dDMs, nchan, fch1, df):
    """
    calc time offset from incorrect dm
    """
    fchans = np.arange(nchan) * df + fch1
    fhi = np.max(fchans)
    
    avg_f2 = np.mean( fchans**-2.0 - fhi**-2.0) 

    dts = 4.15e3 * avg_f2 * dDMs

    return dts


def sift_dm_dupes(clist, nchan, fch1, df, dt):
    """
    sift through candidate list and remove 
    candidates that may be duplicates of 
    higher SNR burts at wrong DMs
    """ 
    snrs = np.array([ cc.snr for cc in clist ])
    xx = np.argsort(snrs)[::-1]

    c_snrs = clist[xx]
    c_out  = []

    while(len(c_snrs)):
        #print(c_snrs[0].snr)
        # add current candidate to output
        c_out.append(c_snrs[0])

        # get dms and times
        dms = np.array([ cc.dm for cc in c_snrs ])
        tts = np.array([ cc.time for cc in c_snrs ])
        wws = np.array([ cc.wbins for cc in c_snrs ]) * dt

        # get dm and time of current candidate 
        dm0 = c_snrs[0].dm
        tt0 = c_snrs[0].time
        ww0 = c_snrs[0].wbins * dt

        # Calc offsets
        dts = t_dm_shifts(dms-dm0, nchan, fch1, df)

        # what cands are within dts?
        cond_xx = (np.abs(tt0-tts) <= np.abs(dts) + (wws + ww0)) & \
                  (np.sign(tt0-tts) * np.sign(dts) >= 0) 

        # cands outside are then just the negation
        yy = np.where( cond_xx == False )[0]

        # Now update the c_snrs array to just keep the 
        # ones that are NOT duplicates 
        c_snrs = c_snrs[yy]

        # print the length as a check
        #print(len(c_snrs))

    # Convert output to array 
    c_out = np.array(c_out)

    return c_out
          


################################
## Cross Matching Cand Lists  ##
################################

def cross_check_clists(clist1, clist2, nchans, fch1, df, dt):
    """
    Check for cross-matches for clist1 in clist2
    """
    # Time Samples
    ts1 = np.array([ cc.samp for cc in clist1 ]) * dt 
    ts2 = np.array([ cc.samp for cc in clist2 ]) * dt 

    # Widths 
    w1 = np.array([ cc.wbins for cc in clist1 ]) * dt 
    w2 = np.array([ cc.wbins for cc in clist2 ]) * dt

    # dms 
    dm1 = np.array([ cc.dm for cc in clist1 ]) 
    dm2 = np.array([ cc.dm for cc in clist2 ]) 

    # match lists 
    midx1 = []
    midx2 = []

    for ii in range(len(clist1)):
        tdiff = ts1[ii] - ts2
        jj = np.argmin( np.abs( tdiff ) )
        td_ij = tdiff[jj]
        dm_dt = t_dm_shifts(dm2[jj]-dm1[ii], nchans, fch1, df)
        cond1 = (np.abs(td_ij) <= np.abs(dm_dt) + (w2[jj] + w1[ii])) & \
                (np.sign(td_ij) * np.sign(dm_dt) >= 0)
        cond2 = np.abs(td_ij) <= (w2[jj] + w1[ii])

        cond = cond1 | cond2

        print(ii, td_ij * 1e3, w2[jj] + w1[ii], dm_dt * 1e3, cond)
        if cond:
            midx1.append(ii)
            midx2.append(jj)
        else:
            pass

    midx1 = np.array(midx1)
    midx2 = np.array(midx2)

    return midx1, midx2
             
        
def cross_check_clists_orig(clist1, clist2):
    """
    Check for cross-matches for clist1 in clist2
    """
    # Samples 
    ts1 = np.array([ cc.samp for cc in clist1 ]) 
    ts2 = np.array([ cc.samp for cc in clist2 ]) 

    # Widths 
    w1 = np.array([ cc.wbins for cc in clist1 ])  
    w2 = np.array([ cc.wbins for cc in clist2 ])  

    # match lists 
    midx1 = []
    midx2 = []

    for ii in range(len(clist1)):
        tdiff = ts1[ii] - ts2
        jj = np.argmin( np.abs( tdiff ) )
        td_ij = np.abs(tdiff[jj])
        if td_ij <= (w1[ii] + w2[jj])//2:
            midx1.append(ii)
            midx2.append(jj)
        else:
            pass

    midx1 = np.array(midx1)
    midx2 = np.array(midx2)

    return midx1, midx2



########################
##  Candidate Params  ##
########################

def cand_params(candlist):
    """
    return arrays of cand pars
    """
    tts = np.array([ cc.time for cc in candlist ])
    dms = np.array([ cc.dm for cc in candlist ])
    wws = np.array([ cc.wbins for cc in candlist ])
    snrs = np.array([ cc.snr for cc in candlist ])

    return tts, snrs, dms, wws
         


#####################
## YOUR Candidates ##
#####################

def sp_to_your_cand(sp_cand, data_file, zap_chans=None, 
                    label=-1, min_samp=256, device=0):
    """
    Convert an SP_CAND object to a your Candidate obect
    
    zap_chans = list of channel numbers (data file order)
                to zero out when de-dispersing + making 
                dm time plots
    """
    cand = Candidate(fp=data_file,
                     dm=sp_cand.dm,
                     tcand=sp_cand.time,
                     width=sp_cand.wbins,
                     label=label,
                     snr=sp_cand.snr,
                     min_samp=min_samp,
                     device=device
                    )

    if zap_chans is not None:
        # Make sure chans are ints
        zz = zap_chans.astype('int')
        
        # Make kill mask array (true = flag)
        kill_mask = np.zeros(cand.nchans, dtype='bool')
        kill_mask[zz] = True
        cand.kill_mask = kill_mask
    return cand


def cand_data(cand, tstart=None, tstop=None):
    """
    Generate various data products and save your candidate 
    to h5 format 
    """
    # Get data chunk from inifle
    print("Reading candidate data")
    cand.get_chunk(tstart=tstart, tstop=tstop)

    # Make dm time data
    print("Making DM vs Time data")
    cand.dmtime()
    
    # Dedispersing
    print("Dedispersing candidate data")
    cand.dedisperse() 

    # Save candidate 
    cand.dm_opt = -1
    cand.snr_opt = -1

    return cand


def save_plot(h5file, detrend=False):
    """
    Make and save a cand plot from h5 file
    """
    plt.ioff()
    plot_h5(h5file, detrend_ft=detrend, save=True)
    plt.close()
    plt.ion()
    return 


def get_start_stop(cand, n):
    """
    Get start / stop times for data chunk

    Will take data half-length to be the larger of:

      n * max(width // 2, 1) * tsamp 

    or 

      (dispersion sweep) + width * tsamp  
    """
    w = cand.width 
    dt = cand.native_tsamp 
    tdm = cand.dispersion_delay() 
    tcand = cand.tcand

    tdur1 = n * max(w//2, 1) * dt
    tdur2 = tdm + w * dt 

    tdur = max( tdur1, tdur2 )

    tstart = tcand - tdur
    tstop  = tcand + tdur

    return tstart, tstop 


def make_h5_from_sp(sp, data_file, snr_min=7, zap_chans=None, 
                    t_dec=-1, f_dec=1, min_samp=0, n=-1, ctype='sp', 
                    extra_filter=True):
    """
    Make h5 candidates from spfile or candlist.  
    Only save candidates with snr > snr_min

    t_dec == - 1 means use width//2
    """
    # Check if candlist or sp file 
    
    # Assuming string is file name  
    if type(sp) == str:
        clist_all = cands_from_spfile(sp, ctype)

    # If list or array assume it is clist
    elif type(sp) in [np.ndarray, list]:
        clist_all = [ cc for cc in sp ] 

    # Otherwise we dont know
    else: 
        print("Unknown sp type: must be file name or array/list")
        return

    # Do extra filtering if needed
    if extra_filter:
        yr = your.Your(data_file)
        foff = yr.your_header.foff
        fch1 = yr.your_header.fch1
        dt   = yr.your_header.tsamp
        nchans = yr.your_header.nchans

        csift = sift_dm_dupes(clist_all, nchans, fch1, foff, dt)
        clist_all = csift

    # Make new array with snr_min cut
    clist = np.array([ cc for cc in clist_all if cc.snr > snr_min ])

    # Print number of cands
    print("Processing %d cands" %len(clist))

    # now convert to your format and save
    for cc in clist:
        # convert to your format 
        yc = sp_to_your_cand(cc, data_file, zap_chans=zap_chans, 
                             min_samp=min_samp)

        # if n > 0, get tstart, tstop otherwise none
        if n > 0:
            tstart, tstop = get_start_stop(yc, n)
        else:
            tstart = tstop = None
            
        yc = cand_data(yc, tstart, tstop)

        # Decimate if desired
        t_dec_fac = t_dec
        f_dec_fac = f_dec
 
        if t_dec == -1:
            t_dec_fac = max(1, yc.width // 2)

        if t_dec_fac > 1 or f_dec_fac > 1:
            print(t_dec_fac, f_dec_fac)
            yc = decimate_cand(yc, t_dec_fac, f_dec_fac)
        else: 
            pass

        # Save data 
        fout = yc.save_h5()
    return 


def decimate_cand(cand, time_fac, freq_fac):
    """
    avg in time and freq 
    """
    # Decimate time axis of tf data
    cand.decimate(key="ft", axis=0, pad=True, 
                  decimate_factor=time_fac, mode="median")

    # Decimate freq axis of tf data
    cand.decimate(key="ft", axis=1, pad=True, 
                  decimate_factor=freq_fac, mode="median")

    # Decimate dmt data
    cand.decimate(key="dmt", axis=1, pad=True, 
                  decimate_factor=time_fac, mode="median")

    return cand


def write_cands(clist, outfile):
    """
    Write out candidates
    """
    with open(outfile, 'w') as fout:
        for cc in clist:
            ostr = "%.1f   %.1f   %.3f    %d   %d\n" %(\
                    cc.dm, cc.snr, cc.time,  cc.samp, cc.wbins)
            fout.write(ostr)

    return


def parse_input():
    """
    Use 'your' to make pulse candidates + plots
    """
    prog_desc = "Make plots of SP cand file"
    parser = ArgumentParser(description=prog_desc)

    parser.add_argument('infile', help='Input *.singlepulse file')
    parser.add_argument('filfile', help='Filterbank file')
    parser.add_argument('-s', '--snr',
                        help='SNR threshold for plotting (default = all)',
                        required=False, type=float, default=-1)
    parser.add_argument('-t', '--tdec',
                        help='Time decimation factor (default = -1, auto)',
                        required=False, type=int, default=-1)
    parser.add_argument('-f', '--fdec',
                        help='Frequency decimation factor (default = 1)',
                        required=False, type=int, default=1)
    parser.add_argument('-n', '--nmin',
                        help='Minimum output time bins (default = 32)',
                        required=False, type=int, default=32)
    args = parser.parse_args()

    infile = args.infile
    filfile = args.filfile
    snr = args.snr
    tdec = args.tdec
    fdec = args.fdec
    nmin = args.nmin

    return infile, filfile, snr, tdec, fdec, nmin

debug = 0

if __name__ == "__main__":
    if debug:
        pass

    else:
        infile, filfile, snr, tdec, fdec, nmin = parse_input()
        
        # Generate *h5 data snippets from cand file
        make_h5_from_sp(infile, filfile, snr_min=snr, 
                        t_dec=tdec, f_dec=fdec, n=nmin, ctype='sp')

        # Make plots from *h5 files using your script
        call("your_h5plotter.py -f *h5", shell=True)
