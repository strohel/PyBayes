#!/usr/bin/env python

import matplotlib.pyplot as plt

x = [2, 30, 60]

pybayes_py = [0.253716135, 0.6892303944, 2.1205345631]
pybayes_cy = [0.0910650969, 0.5349771023, 1.8155289888]
matlab_imper = [0.0690379, 0.4242489, 1.2736401]
matlab_oo = [1.3781965, 1.780366, 3.8488068]
bdm = [0.0255741, 0.5185413, 1.9475628]

ax = plt.figure().add_subplot(111)
plt.plot(x, pybayes_py, '--v', label='PyBayes Py')
plt.plot(x, pybayes_cy, '-o', label='PyBayes Cy')
plt.plot(x, matlab_imper, '-D', label='MATLAB imper.')
plt.plot(x, matlab_oo, '--^', label='MATLAB o-o')
plt.plot(x, bdm, '-s', label='BDM')

plt.legend(loc='upper left')
ax.set_xlabel('number of state-space dimensions')
ax.set_ylabel('total run time [s]')
ax.set_xlim(right=62)

plt.show()
