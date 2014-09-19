'''
Created on Jun 26, 2014

@author: eija
'''

if __name__ == '__main__':
    pass

import numpy as np
import sys
import nrrd
import StringIO

def print_gradients(gradients, suffix):
    print  'GRADIENTS ' + suffix
    for key in gradients:
        print str(key[0]) + ' [' + key[1] + ']:' + str(key[2]) + ',' + str(key[3]) + ',' + str(key[4])

def readNRRDdata(filename):
    readdata, options = nrrd.read(filename)
    gradients = []
    for field in options:
        if field=='keyvaluepairs':
            for key in options[field]:
                if key.find('DWMRI_gradient') != -1:
                    g_str = options[field][key]
                    while g_str.find('  ') != -1:
                        g_str = g_str.replace('  ',' ');
                    g_str = g_str.split(' ')
                    # Parse gradient number from its name
                    g_no_name = int(float(key[-4:]))
                    gradients.append((g_no_name, key, float(g_str[0]), float(g_str[1]), float(g_str[2])))
    return readdata, options, gradients

def cat_options(options, options_new, g_all_i):
    for key in options['keyvaluepairs']:
        if key.find('DWMRI_gradient') != -1:
            buf = StringIO.StringIO()
            buf.write("DWMRI_gradient_%04d" % g_all_i)
            print buf.getvalue() + ':[' + key + ']' + options['keyvaluepairs'][key]
            options_new[buf.getvalue()] = options['keyvaluepairs'][key]
        else:
            options_new[key] = options['keyvaluepairs'][key]
        g_all_i = g_all_i + 1
    return options_new, g_all_i

def dtimerge(filelist,outputname):

    # Collect components to list
    datalist = []
    no_directions = 0
    shape = []
    for file_i in range(len(filelist)):
        data, options, gradients = readNRRDdata(filelist[file_i])
        print_gradients(gradients, str(file_i))
        datalist.append((data,options,gradients))
        no_directions = no_directions + data.shape[3]
        if len(shape) > 0 and (shape[0] != data.shape[0] or shape[1] != data.shape[1] or shape[2] != data.shape[2]):
            raise "shapes did not match"
        shape = data.shape

    # Concatenate data matrix
    shape = (shape[0], shape[1], shape[2], no_directions)
    print shape
    writedata = np.empty(shape)
    g_all_i = 0
    for list_i in range(len(datalist)):
        data = datalist[list_i][0]
        for g_i in range(data.shape[3]):
            writedata[:,:,:,g_all_i] = data[:,:,:,g_i]
            g_all_i = g_all_i + 1

    # Concatenate NRRD header
    writeoptions = datalist[0][1]
    options_new_pairs = {}
    g_all_i = 0
    keyvaluepairs_new = {}
    for list_i in range(len(datalist)):
        options = datalist[list_i][1]
        keyvaluepairs_new, g_all_i = cat_options(options, keyvaluepairs_new, g_all_i)
    writeoptions['keyvaluepairs'] = keyvaluepairs_new

    '''
        Write output
        '''
    nrrd.write(outputname, writedata, writeoptions)
