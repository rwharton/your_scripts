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
        
        #tfix = 4.15e3 * (fref**-2.0 - fhi**-2.0) * dm
        #nfix = int( tfix / dt )
        tfix = 0.0
        nfix = 0

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

def get_chan_info(data_file):
    yr = your.Your(data_file)
    foff = yr.your_header.foff
    fch1 = yr.your_header.fch1
    dt   = yr.your_header.tsamp
    nchans = yr.your_header.nchans
    
    return nchans, fch1, foff, dt


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


def get_csift(clist, data_file):
    """
    Run extra dm sifting to clist
    """
    yr = your.Your(data_file)
    foff = yr.your_header.foff
    fch1 = yr.your_header.fch1  
    dt   = yr.your_header.tsamp
    nchans = yr.your_header.nchans
    csift = sift_dm_dupes(clist, nchans, fch1, foff, dt)
    return csift
    


def make_h5_from_sp(sp, data_file, snr_min=7, wmax=-1, zap_chans=None, 
                    t_dec=-1, f_dec=1, min_samp=0, n=-1, ntmax=-1, 
                    ctype='sp', extra_filter=True):
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
    
    # Make new array with wmax cut
    if wmax <= 0:
        pass
    else:
        clist = np.array([ cc for cc in clist if cc.wbins <= wmax ])

    # Print number of cands
    print("Processing %d cands" %len(clist))

    # now convert to your format and save
    Nc = len(clist)
    for ii, cc in enumerate(clist):
        print("Cand %d/%d" %(ii, Nc))
        # convert to your format 
        yc = sp_to_your_cand(cc, data_file, zap_chans=zap_chans, 
                             min_samp=min_samp)

        # if n > 0, get tstart, tstop otherwise none
        if n > 0:
            tstart, tstop = get_start_stop(yc, n)
        else:
            tstart = tstop = None
            
        yc = cand_data(yc, tstart, tstop)

        # Fix annoying null char issues in header
        yc.source_name = yc.source_name.rsplit(b'\x00')[0]
        yc.your_header.source_name =\
                    yc.your_header.source_name.rsplit('\x00')[0]

        # Decimate if desired
        t_dec_fac = t_dec
        f_dec_fac = f_dec
 
        if t_dec == -1:
            t_dec_fac = max(1, yc.width // 2)

        if t_dec_fac > 1 or f_dec_fac > 1 or ntmax > 0:
            print(t_dec_fac, f_dec_fac, ntmax)
            #yc = decimate_cand(yc, t_dec_fac, f_dec_fac)
            yc = decimate_crop_cand(yc, t_dec_fac, f_dec_fac, ntmax)
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


def decimate_crop_cand(cand, time_fac, freq_fac, ntsamp=-1):
    """
    avg in time and freq 

    crop in time if required and desired
    """
    # Decimate time axis of tf data
    cand.decimate(key="ft", axis=0, pad=True, 
                  decimate_factor=time_fac, mode="median")

    # Crop in time if desired and nec
    dd_nt = cand.dedispersed.shape[0] 
    if ntsamp > 0 and ntsamp < dd_nt:
        crop_ft_start = dd_nt // 2 - ntsamp // 2
        cand.dedispersed = crop(cand.dedispersed, crop_ft_start, ntsamp, 0)

    # Decimate freq axis of tf data
    cand.decimate(key="ft", axis=1, pad=True, 
                  decimate_factor=freq_fac, mode="median")

    # Decimate dmt data
    cand.decimate(key="dmt", axis=1, pad=True, 
                  decimate_factor=time_fac, mode="median")

    # crop in time if desired and nec
    dmt_nt = cand.dmt.shape[1]
    if ntsamp > 0 and ntsamp < dmt_nt:
        crop_dmt_start = dmt_nt // 2 - ntsamp // 2
        cand.dmt = crop(cand.dmt, crop_dmt_start, ntsamp, 1)

    if time_fac > 1:
        dt = cand.tsamp
        cand.tsamp = dt * time_fac

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


#######################
###  SUMMARY PLOTS  ###
#######################

def time_summary(csift, clist, title=None, outfile=None):
    """
    Summary plots along time axis
    """
    if outfile is not None:
        plt.ioff()
    else: pass

    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    tt0, snr0, dm0, ww0 = cand_params(clist)
    tt1, snr1, dm1, ww1 = cand_params(csift)

    # Top Panel is DM vs time with SNR size
    ax1.scatter(tt0, dm0, s=25*(snr0/10)**2.0, fc='none', 
                ec='k', lw=1)
    ax1.scatter(tt1, dm1, s=25*(snr1/10)**2.0, ec='r', 
                marker='s', lw=1, fc='none')

    ax1.set_yscale('log')
    ax1.set_ylabel('DM (pc/cc)', fontsize=14)

    # Middle panel is count rate per minute
    bb = np.arange(0, np.max(tt0), 60)
    ax2.hist(tt0, bins=bb, color='k')
    ax2.hist(tt1, bins=bb, color='r')

    ax2.set_yscale('log')
    ax2.set_ylabel('Hits/min', fontsize=14)

    # Bottom plot is snr
    ax3.plot(tt0, snr0, marker='.', ls='', c='k')
    ax3.plot(tt1, snr1, marker='o', mfc='none', mec='r', 
             ls='')
    ax3.set_ylim(5)

    ax3.set_ylabel('SNR', fontsize=14)
    ax3.set_xlabel("Time (s)", fontsize=14)
    
    if title is not None:
        fig.suptitle(title, fontsize=16, y=0.925)

    if outfile is not None:
        plt.savefig(outfile, dpi=100, bbox_inches='tight')
        plt.close()
        plt.ion()
    else:
        plt.show()
    
    return


def snr_summary(csift, clist, title=None, outfile=None):
    """
    Summary plots along time axis
    """
    if outfile is not None:
        plt.ioff()
    else: pass
    
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    tt0, snr0, dm0, ww0 = cand_params(clist)
    tt1, snr1, dm1, ww1 = cand_params(csift)

    fs = 10

    # Top left plot is SNR Hist
    ax1.hist(snr1, log=True)
    ax1.set_xlabel("SNR", fontsize=fs)

    # Top middle plot is DM hist
    dbins = 10**np.linspace(2, 6, 20)
    ax2.hist(dm1, bins=dbins)
    ax2.set_xscale('log') 
    ax2.set_xlabel("DM (pc/cc)", fontsize=fs)

    # Top right plot is width (bins)
    wbins = 2**np.arange(10) / np.sqrt(2)
    ax3.hist(ww1, bins=wbins)
    ax3.set_xscale('log', base=2)
    ax3.set_xlabel("Width (bins)", fontsize=fs)
    

    # Bottom Left plot DM vs Width
    ax4.plot(dm1, ww1, marker='o', ls='', mfc='none')
    ax4.set_xscale('log', base=10)
    ax4.set_yscale('log', base=2)
    ax4.set_xlabel('DM (pc/cc)', fontsize=fs)
    ax4.set_ylabel('Width (bins)', fontsize=fs)

    # Bottom Middle plot DM vs SNR
    ax5.plot(dm1, snr1, marker='o', ls='', mfc='none')
    ax5.set_xscale('log')
    ax5.set_xlabel('DM (pc/cc)', fontsize=fs)
    ax5.set_ylabel('SNR', fontsize=fs)

    # Bottom Right plot width vs SNR
    ax6.plot(ww1, snr1, marker='o', ls='', mfc='none')
    ax6.set_xscale('log', base=2)
    ax6.set_xlabel('Width (bins)', fontsize=fs)
    ax6.set_ylabel('SNR', fontsize=fs)

    if title is not None:
        fig.suptitle(title, fontsize=16, y=0.925)
    
    if outfile is not None:
        plt.savefig(outfile, dpi=100, bbox_inches='tight')
        plt.close()
        plt.ion()
    else:
        plt.show()
    
    return


def make_summary_plots(spfile, data_file, basename, 
                       ctype='cand'):
    """
    make and save the two summary plots

    basename will be the title of each plot 
    and the base of the outfile name
    """
    clist = cands_from_spfile(spfile, ctype=ctype)
    csift = get_csift(clist, data_file)
    
    # Time summary
    time_file = "%s_time.png" %basename
    time_summary(csift, clist, title=basename, 
                 outfile=time_file)

    # SNR summary 
    snr_file = "%s_snr.png" %basename
    snr_summary(csift, clist, title=basename, 
                outfile=snr_file)

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
    parser.add_argument('-m', '--nmax',
                        help='Maximum output time bins (default = -1 ie no max)',
                        required=False, type=int, default=-1)
    parser.add_argument('-w', '--wmax',
                        help='Maximum width to consider (default = -1 ie no max)',
                        required=False, type=int, default=-1)
    parser.add_argument('--no_cands', action='store_true', 
                        help='Do NOT make candidate plots (default false)', 
                        default=False)
    parser.add_argument('--no_sum', action='store_true', 
                        help='Do NOT make summary plots (Default false)', 
                        default=False)
    parser.add_argument('-b', '--basename',
                        help='Basename for summary plots (default = cand)',
                        required=False, default="cand")
    parser.add_argument('-c', '--ctype',
                        help='Input cand type (0 = sp, 1 = heimdall) (Default=1)',
                        required=False, type=int, default=1)
    
    



    args = parser.parse_args()

    infile = args.infile
    filfile = args.filfile
    snr = args.snr
    tdec = args.tdec
    fdec = args.fdec
    nmin = args.nmin
    nmax = args.nmax
    wmax = args.wmax
    skip_cands = args.no_cands
    skip_sum = args.no_sum
    bname = args.basename
    cnum = args.ctype

    return infile, filfile, snr, tdec, fdec, nmin, nmax, wmax, skip_cands, skip_sum, bname, cnum

debug = 0

if __name__ == "__main__":
    if debug:
        pass

    else:
        infile, filfile, snr, tdec, fdec, nmin, nmax, wmax, skip_cands, skip_sum, bname, cnum = parse_input()

        ctype = 'cand'
        if cnum == 0:
            ctype = 'sp'

        # Make summary plots
        if not skip_sum:
            make_summary_plots(infile, filfile, bname, ctype)
        
        # Generate *h5 data snippets from cand file
        # and make plots from *h5 files
        if not skip_cands:
            make_h5_from_sp(infile, filfile, snr_min=snr, wmax=wmax, 
                            t_dec=tdec, f_dec=fdec, n=nmin, ntmax=nmax, 
                            ctype=ctype)
            call("your_h5plotter.py -f *h5", shell=True)

