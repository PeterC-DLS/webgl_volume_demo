import logging

import numpy as np
import h5py
import nrrd

"""
    processed/reciprocal_space:NXdata/
                                       @auxiliary_signals='weight'
                                       @axes=['h-axis', 'k-axis', 'l-axis']
                                       @h-axis_indices=0
                                       @k-axis_indices=1
                                       @l-axis_indices=2
                                       @signal='volume'
    processed/reciprocal_space/h-axis: (64,) units=/angstrom
    processed/reciprocal_space/k-axis: (96,) units=/angstrom
    processed/reciprocal_space/l-axis: (96,) units=/angstrom
    processed/reciprocal_space/volume: (64, 96, 96)
    processed/reciprocal_space/weight: (64, 96, 96)
"""
def load_vol_nxs(nxs, group="processed/reciprocal_space", data_name="volume", axis_names=None):
    """Load volume data

    If a NeXus file is given and the group is an NXdata, then the signal data and its axes are discovered so data_name and axis_names

    Parameters
    ----------
    nxs : _type_
        file path to NeXus/hdf5 file
    group : str, optional
        path to group (could be NXdata) containing volume and axes, by default "processed/reciprocal_space"
    data_name : str, optional
        name of data containing volume, by default "volume"
    axis_names : tuple(str), optional
        names of axis data

    Returns
    -------
    _type_
        _description_
    """
    with h5py.File(nxs) as f:
        def_entry = f.attrs.get("default")
        if def_entry is not None: # attempt to get default NXdata
            def_entry = def_entry.decode()
            entry = f.get(def_entry)
            if entry is None:
                logging.warning("Default entry (%s) not found", def_entry)
            else:
                def_data = entry.attrs.get("default")
                if def_data is None:
                    logging.warning("default attribute not found in %s", def_entry)
                else:
                    def_data = def_data.decode()
                    if def_data not in entry:
                        logging.warning("default NXdata (%s) not found in %s", def_data, tuple(entry.keys()))
                    else:
                        group = f"{def_entry}/{def_data}"

        g = f[group]
        vol_data = {}
        volume = None
        axes_attr = None
        nx_class = g.attrs.get("NX_class")
        if nx_class == b"NXdata":
            logging.warning("Given group is an NXdata")
            signal_name = g.attrs.get("signal", "data")
            volume = g.get(signal_name)
            if volume is None:
                logging.warning("No signal data found in %s", tuple(g.keys()))
            else:
                vol_data["volume"] = volume[...]

            axes_attr = g.attrs.get("axes")
            if axes_attr is None:
                logging.warning("axes attribute missing from %s", group)

        if volume is None:
            volume = g[data_name]
            vol_data["volume"] = volume[...]

        if axes_attr is not None:
            axis_names = [ a.decode() for a in axes_attr ]

        if axis_names is None:
            logging.warning("No axes found or given")
        else:
            vol_data["axes"] = { n:g[n][...] for n in axis_names }
        
        return vol_data


def save_vol_nrrd(nrrd_file, vol_data, out_dtype=np.float32):
    axes = vol_data["axes"]
    spacings = [ np.round(i[1]-i[0],15) for i in axes.values() ]
    origin = [ np.round(i[0],15) for i in axes.values() ]

    header = dict(space="right-anterior-superior", labels=list(axes.keys()), spacings=spacings)
    header["space origin"] = origin
    volume = vol_data["volume"]
    v_min, v_max = volume.min(), volume.max()
    print(volume.shape, v_min, v_max, origin)
    volume = (volume - v_min) * (1 / (v_max - v_min))
    nrrd.write(nrrd_file, volume.astype(out_dtype), header)

def save_vol_npz(npz_file, vol_data, out_dtype=np.float32):
    np.savez(npz_file, **vol_data)

if __name__ == "__main__":
    import sys
    argv = sys.argv
    argc = len(argv)
    if argc < 2:
        raise ValueError("Need an input file")
    is_nrrd = "nrrd" in argv[0]
    in_name = argv[1]
    in_dict = load_vol_nxs(in_name)
    if argc > 2:
        out_name = argv[2]
    else:
        idx = in_name.rfind(".")
        out_name = (in_name[:idx] if idx > 0 else in_name) + (".nrrd" if is_nrrd else ".npz")

    if is_nrrd:
        save_vol_nrrd(out_name, in_dict)
    else:
        save_vol_npz(out_name, in_dict)

