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
tickrotation=0
width = 0.3

# from 02-14-2019_22.47.54
data = {
    'Seq.': {
        'Sequential': [400591.17, 400919.34, 400640.22],
    },
    'Row on\nGPU': {
        'CUDA': [129099.79, 129337.19, 129336.09],
        'OpenCL': [129817.99, 129613.99, 129624.76],
    },
    'Row on GPU\nwith 2D grid': {
        'CUDA': [252181.75, 251433.35, 251407.90],
        'OpenCL': [247959.76, 247560.01, 247538.42],
    },
    'Batch of\n32 rows': {
        'CUDA': [8877.72, 8925.50, 8882.12, 8913.03, 9095.67, 8937.54, 9007.57, 8979.65, 8972.49],
        'OpenCL': [9052.18, 9060.56, 9207.89, 9129.99, 9222.14, 9250.51, 9084.33, 9137.35, 9139.05],
    },
    '2x mem.\nspaces': {
        'CUDA': [5975.43, 6021.51, 5933.81, 5958.19],
        'OpenCL': [5974.40, 5960.01, 5922.51, 6084.24],
    },
    '4x mem.\nspaces': {
        'CUDA': [5413.96, 5437.52, 5470.52, 5452.26],
        'OpenCL': [5383.14, 5375.73, 5409.21, 5459.63],
    },
    '2 GPUs': {
        'CUDA': [4468.37, 4467.62, 4531.85, 4480.11],
        'OpenCL': [4494.29, 4526.30, 4457.13, 4451.80],
    },
    '2 GPUs with\n2x memory': {
        'CUDA': [3075.11, 3036.97, 3010.60, 2991.93],
        'OpenCL': [3079.83, 3036.72, 3095.13, 3096.35],
    },
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
i = 0
w_i = 0
for env in environments:
    v_i = 0
    for ver in versions:
        if env in data and ver in data[env]:
            value = np.mean(data[env][ver])
            print('%s %s: %.2f ms' % (env, ver, value))
            barcontainer = ax1.bar(
                pos,
                value,
                label=ver,
                yerr=np.std(data[env][ver]),
                width=width,
                bottom=0,
                error_kw={'elinewidth': 1, 'capsize': 3},
                color=colors[v_i],
                edgecolor='w',
                hatch=patterns[v_i] if patterns else None,
            )
            patches[v_i] = barcontainer.patches[0]

            speedup['val'].append(seq_time/value)
            speedup['pos'].append(pos)

            i += 1
            pos += width

        v_i += 1
    
    pos += width
    w_i += 1
    i += 1


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
ax1.set_xticklabels(environments, rotation=tickrotation)

ax1.set_yscale('log')

ax1.legend(patches,
    versions,
    title=None,
    loc="upper center",
    # loc=(0.2,-0.15),
    ncol=4,
    fancybox=False,
    handlelength=3,
    handleheight=1.2,
    # bbox_to_anchor=(0.5, 0, 0.5, 1)
)

fig1.set_size_inches(8, 6)

filename = splitext(basename(__file__))[0]
for fmt in ('png','pdf'): #png, pdf, ps, eps, svg
    plt.savefig(fmt + '/' + filename + '.' + fmt, dpi=100, format=fmt, bbox_inches='tight')
# plt.show()
plt.close()
