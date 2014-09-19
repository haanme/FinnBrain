#!/usr/bin/env python

#
# Moves file to results folder, overwriting the existing file
#
# filename   - file to be moved
# out_prefix - subject specific prefix
#
def move_to_results(filename, out_prefix):
    import os
    import shutil
    outfile = experiment_dir + '/' + out_prefix + '/' + os.path.basename(filename)
    if os.path.isfile(outfile):
        os.remove(outfile)
    shutil.move(filename,outfile)
    return outfile

#
# Copies file to results folder, overwriting the existing file
#
# filename     - file to be moved
# new_filename - new filename
# out_prefix   - subject specific prefix
#
def copy_to_results(filename, new_filename, out_prefix):
    import os
    import shutil
    outfile = experiment_dir + '/' + out_prefix + '/' + new_filename
    if os.path.isfile(outfile):
        os.remove(outfile)
    print "copying [" + filename + "]->[" + outfile + "]"
    shutil.copy(filename,outfile)
    return outfile

#
# Split filename into (<root>/<basename>.<extension>)
#
# filename - filename that is splitted
#
def split_nii_gz(filename):
    import os
    # split path and filename
    root, basename = os.path.split(filename)
    # split extensions until none is found, catenating extensions
    basename, ext_new = os.path.splitext(basename)
    ext = ext_new
    while len(ext_new) > 0:
        basename, ext_new = os.path.splitext(basename)
        ext = ext_new + ext
    return root, basename, ext

#
# Renames file basename in its location
#
# filename     - filename that is renamed
# basename_new - new basename
#
def rename_basename_to(filename, basename_new):
    import os
    import shutil
    root, basename, ext = split_nii_gz(filename)
    filename_new = os.path.join(root, (basename_new + ext))
    if os.path.isfile(filename_new):
        os.remove(filename_new)
    shutil.move(filename,filename_new)
    return filename_new

