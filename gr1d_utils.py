import numpy as np
import h5py 
import os
import re

time_rexp = re.compile("Time =[\s]*([\d.\d]*)")
val_pair_rexp = re.compile("[\s]*([\d.E+-]*)[\s]*([\d.E+-]*)")
varname_rexp = re.compile("([\S]*).xg")

req_vars_for_ETK = ["rho","ye","eps","alpha","volume","v","v1","temperature","press"]

def extract_ETK_data(in_fname,varnames=req_vars_for_ETK,out_time="final",out_fname="dat_ETK.h5"):
    vars_ = []
    with h5py.File(in_fname,"r") as f:
        t = f["t"][:]
        if out_time=="final": indt_ = -1
        else: indt=np.searchsorted(t,out_time)
        t = t[indt]
        x = f["r"][:]
        ind = get_linear_grid_extent(x)
        x = x[:ind]
        for vname_ in varnames:
            vars_.append((vname_,f[vname_][indt,:ind]))

    write_var_1d(t,"t",out_fname)
    write_var_1d(x,"r",out_fname)
    for (vname_,var_) in vars_:
        write_var_2d(var_,vname_,out_fname)
    return

def gr1d_save_hdf5_database(basedir,varnames=None,fname="dat.h5"):
    if not os.path.isdir(basedir):
        raise ValueError("Non existent directory!")
    if os.path.isfile(fname):
        os.remove(fname)
    if varnames is None:
        varnames=[]
        ls_ = os.listdir(basedir)
        for item in ls_:
            match_ = varname_rexp.match(item)
            if match_: varnames.append(match_.group(1))
    vname_ = "rho"
    xg_fname =  os.path.join(basedir,vname_+".xg")
    t,x,v = read_xg_file(xg_fname)
    write_var_1d(t,"t",fname)
    write_var_1d(x,"r",fname)
    for vname_ in varnames:
        xg_fname = os.path.join(basedir,vname_+".xg")
        _,_,v = read_xg_file(xg_fname)
        write_var_2d(v,vname_,fname)
    return

def read_xg_file(fname):
    if not os.path.isfile(fname):
        raise ValueError("Filename provided doesn't exist")
    with open(fname) as f:
        timesteps = f.read().split('"')[1:]
    t = np.zeros(len(timesteps))
    t_,x,var_ = read_xg_timestep(timesteps[0])
    var = np.zeros((len(timesteps),len(x)))
    for i,ts in enumerate(timesteps):
        t_,x,var_ = read_xg_timestep(ts)
        t[i] = t_
        var[i,:] = var_
    return t,x,var
    
def read_xg_timestep(ts):
    match_ = time_rexp.match(ts)
    t = float(match_.group(1))
    tts = ts.split("\n")[1:-3]
    x = np.zeros(len(tts))
    var = np.zeros_like(x)
    for i,line in enumerate(tts):
        match_ = val_pair_rexp.match(line)
        x[i] = float(match_.group(1))
        var[i] = float(match_.group(2))
    return t,x,var

def write_var_2d(var,varname,fname):
    if not os.path.isfile(fname):
        raise ValueError("Filename provided doesn't exist, this routine shouldn't be creating it!")
    f = h5py.File(fname,"r+")
    f.create_dataset(varname,data=var)
    f.close()
    return

def write_var_1d(var,varname,fname):
    f = h5py.File(fname,"a")
    f.create_dataset(varname,data=var)
    f.close()
    return 

def get_linear_grid_extent(x):
    dx = np.diff(x)
    for i in np.arange(len(dx)-1,1,-1):
        if not(dx[i-1]==dx[i]): index=i
    return index
