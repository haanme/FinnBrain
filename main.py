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

data1, options1, gradients1 = readNRRDdata('../20140505_155648ep2ddifftensor34pat2s016a001NRRD.nrrd')
data2, options2, gradients2 = readNRRDdata('../20140505_155648ep2ddifftensor35pat2s022a001NRRD.nrrd')
data3, options3, gradients3 = readNRRDdata('../20140505_155648ep2ddifftensor36pat2s028a001NRRD.nrrd')
print_gradients(gradients1, '1')
print_gradients(gradients1, '2')
print_gradients(gradients1, '3')

# Concatenate data matrix
shape = (data1.shape[0], data1.shape[1], data1.shape[2], data1.shape[3]+data2.shape[3]+data3.shape[3])
print shape
writedata = np.empty(shape)
g_all_i = 0
for g_i in range(data1.shape[3]):
    writedata[:,:,:,g_all_i] = data1[:,:,:,g_i]
    g_all_i = g_all_i + 1
for g_i in range(data2.shape[3]):
    writedata[:,:,:,g_all_i] = data2[:,:,:,g_i]
    g_all_i = g_all_i + 1
for g_i in range(data3.shape[3]):
    writedata[:,:,:,g_all_i] = data3[:,:,:,g_i]
    g_all_i = g_all_i + 1

writeoptions = options1
options_new_pairs = {}
g_all_i = 0
keyvaluepairs_new = {}
keyvaluepairs_new, g_all_i = cat_options(options1, keyvaluepairs_new, g_all_i)
keyvaluepairs_new, g_all_i = cat_options(options2, keyvaluepairs_new, g_all_i)
keyvaluepairs_new, g_all_i = cat_options(options3, keyvaluepairs_new, g_all_i)
writeoptions['keyvaluepairs'] = keyvaluepairs_new

'''
Write output
'''
nrrd.write('merged.nrrd', writedata, writeoptions)
