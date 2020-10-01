import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates

# Fake data using dates as requested
xdata = np.array([datetime.datetime.today()+
                 datetime.timedelta(days=1)*i for i in range(15)])
ydata = np.cumsum(np.random.uniform(size=len(xdata)))
xlims = mdates.date2num([xdata[0], xdata[-1]])

# Construct an image linearly increasing in y
xv, yv = np.meshgrid(np.linspace(0,1,50), np.linspace(0,1,50))
zv = yv

# Draw the image over the whole plot area
fig, ax = plt.subplots(figsize=(5,3))
ax.imshow(zv, cmap='YlGnBu_r', origin='lower',
          extent=[xlims[0], xlims[1], ydata.min(), ydata.max()])

# Erase above the data by filling with white
ax.fill_between(xdata, ydata, ydata.max(), color='w')

# Make the line plot over the top
ax.plot(xdata, ydata, 'b-', linewidth=2)

ax.set_ylim(ydata.min(), ydata.max())
fig.autofmt_xdate()

plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# get colormap
ncolors = 256
color_array = plt.get_cmap('gist_rainbow')(range(ncolors))

# change alpha values
color_array[:,-1] = np.linspace(1.0,0.0,ncolors)

# create a colormap object
map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)

# register this new colormap with matplotlib
plt.register_cmap(cmap=map_object)

# show some example data
f,ax = plt.subplots()
h = ax.imshow(np.random.rand(100,100),cmap='rainbow_alpha')
plt.colorbar(mappable=h)
plt.show()

