#!/usr/bin/env python3

from os.path import basename,splitext
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

# {'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
patterns = ('//', '+', 'x', '.', '\\', 'o', '-', '*')
patterns = patterns*5 #to repeat patterns
light_colors = True
legend = False
title = None
legend_location = (0,-0.23)
legend_location_troughput = (-0.05,-0.23)
width = 0.3
pixels = 2000 * 2000 #image resolution

# from 02-17-2019_03.38.47
data = {
    'CPU-only': {
        'Sequential': [419429.69, 419536.89, 419699.08, 419908.80, 419913.11, 419993.10],
        'SPar': [24924.27, 24896.45, 24901.41, 24899.36],
        'FastFlow': [24899.18, 24873.04, 24872.79, 24866.43],
        'TBB': [24576.18, 24563.31, 24555.66, 24560.00, 24558.39, 24557.72],
    },
    '1 GPU': {
        'CUDA': [5450.47, 5390.73, 5436.60, 5438.95, 5468.69],
        'OpenCL': [5415.83, 5449.09, 5430.40, 5468.08, 5451.14],
        'SPar + CUDA': [5539.35, 5406.99, 5416.56, 5460.92, 5449.06],
        'FF + CUDA': [5745.15, 5592.43, 5668.42, 5517.17, 5537.78],
        'TBB + CUDA': [5460.18, 5434.60, 5705.41, 5589.31, 5555.69],
        'SPar + OpenCL': [5469.34, 5761.15, 5692.31, 5494.65, 5518.61],
        'FF + OpenCL': [5578.30, 5498.64, 5618.32, 5754.69, 5470.37],
        'TBB + OpenCL': [5559.09, 5484.81, 5432.53, 5539.49, 5557.98],
    },
    '2 GPUs': {
        'CUDA': [2820.71, 2826.53, 2855.45, 2838.17, 2871.69],
        'OpenCL': [2808.50, 2800.40, 2824.84, 2804.95, 2783.32],
        'SPar + CUDA': [2792.07, 2802.31, 2804.05, 2796.16, 2789.17],
        'FF + CUDA': [2783.62, 2790.60, 2798.36, 2793.32, 2789.26],
        'TBB + CUDA': [2797.85, 2797.99, 2794.95, 2801.88, 2825.38],
        'SPar + OpenCL': [2890.86, 2795.32, 2771.64, 2795.75, 2797.41],
        'FF + OpenCL': [2809.36, 2881.45, 2812.60, 2862.72, 2782.28],
        'TBB + OpenCL': [2805.11, 2835.43, 2761.49, 2756.45, 2802.91],
    }
}

environments = list(data.keys())
versions = list()
for env in data:
    for ver in data[env]:
        versions.append(ver)
versions = list(OrderedDict.fromkeys(versions).keys())

seq_time = np.mean(data[environments[0]][versions[0]])

cmap = plt.get_cmap('tab20')
color_range = np.arange(10)*2
if light_colors:
    color_range += 1
colors = cmap(list(color_range)*5)

ind = np.arange(len(environments))

# Start plotting

fig1, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.set_ylabel('Execution time (log(s))')
ax2.set_ylabel('Speedup (x)')

patches = [None] * len(versions)

speedup = {
    'val': list(),
    'pos': list(),
}
pos = 0
for env in environments:
    for ver,ver_i in zip(versions, range(0,len(versions))):
        if env in data and ver in data[env]:
            value = np.mean(data[env][ver])
            barcontainer = ax1.bar(
                pos,
                value,
                label=ver,
                yerr=np.std(data[env][ver]),
                width=width,
                bottom=0,
                error_kw={'elinewidth': 1, 'capsize': 3},
                color=colors[ver_i],
                edgecolor='w',
                hatch=patterns[ver_i] if patterns else None,
            )
            patches[ver_i] = barcontainer.patches[0]

            speedup['val'].append(seq_time/value)
            speedup['pos'].append(pos)

            pos += width
    
    pos += width


ax2.plot(speedup['pos'], speedup['val'], 'r.:')

for pos,val in zip(speedup['pos'], speedup['val']):
    ax2.annotate(
        "%.1f" % val if val < 10 else "%.0f" % val,
        (pos, val),
        va='bottom',
        ha='center',
        xytext=(0, 3),
        textcoords='offset pixels',
        fontsize='small',
    )
pos = 0
ticks = list()
for env in data:
    thispos = (len(data[env])-1)/2*width
    ticks.append(pos+thispos)
    pos += (len(data[env])+1)*width
ax1.set_xticks(ticks)
ax1.set_xticklabels(environments)

ax1.set_yscale('log')

ax1.legend(patches,
    versions,
    title=None,
    # loc="upper center",
    loc=legend_location,
    ncol=4,
    fancybox=False,
    handlelength=3,
    handleheight=1.2,
    # bbox_to_anchor=(0.5, 0, 0.5, 1)
)

if title:
    plt.title(title)

fig1.set_size_inches(8, 6)

filename = splitext(basename(__file__))[0]
for fmt in ('png','pdf'): #png, pdf, ps, eps, svg
    plt.savefig(fmt + '/' + filename + '.' + fmt, dpi=100, format=fmt, bbox_inches='tight')
# plt.show()
plt.close()


# Throughput Plot

fig1, ax1 = plt.subplots()
ax1.set_ylabel('Throughput (pixels/s)')

patches = [None] * len(versions)

pos = 0
for env in environments:
    for ver,ver_i in zip(versions, range(0,len(versions))):
        if env in data and ver in data[env]:
            data_seconds = np.divide(data[env][ver], 1000)
            throughput = np.divide(pixels, data_seconds)
            value = np.mean(throughput)
            stderr = np.std(throughput)
            barcontainer = ax1.bar(
                pos,
                value,
                label=ver,
                yerr=stderr,
                width=width,
                bottom=0,
                error_kw={'elinewidth': 1, 'capsize': 3},
                color=colors[ver_i],
                edgecolor='w',
                hatch=patterns[ver_i] if patterns else None,
            )
            patches[ver_i] = barcontainer.patches[0]

            pos += width
    
    pos += width


pos = 0
ticks = list()
for env in data:
    thispos = (len(data[env])-1)/2*width
    ticks.append(pos+thispos)
    pos += (len(data[env])+1)*width
ax1.set_xticks(ticks)
ax1.set_xticklabels(environments)

ax1.legend(patches,
    versions,
    title=None,
    # loc="upper center",
    loc=legend_location_troughput,
    ncol=4,
    fancybox=False,
    handlelength=3,
    handleheight=1.2,
    # bbox_to_anchor=(0.5, 0, 0.5, 1)
)

if title:
    plt.title(title)

fig1.set_size_inches(8, 6)

filename = splitext(basename(__file__))[0]
for fmt in ('png','pdf'): #png, pdf, ps, eps, svg
    plt.savefig(fmt + '/' + filename + '_throughput.' + fmt, dpi=100, format=fmt, bbox_inches='tight')
# plt.show()
plt.close()