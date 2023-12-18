from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
plt.rcParams['figure.figsize'] = [9, 9]

#setup the plotting axes
fig, axs = plt.subplots(nrows=2, ncols=2)

def read_fits(file_path):
    with fits.open(file_path) as hdul:
        data = hdul[0].data
    return data

def create_pixel_matrix(fits_files):
    data_matrix = []
    for fits_file in fits_files:
        data = read_fits(fits_file)
        data_matrix.append(data)
    return np.array(data_matrix)

#create pixel matrix of image
fits_files = ["Jupiter/Jupiter_Light11-21-8.fits"]
pix_matrix = np.squeeze(create_pixel_matrix(fits_files))

#normalize pixels and get shape of image
pix_matrix = (pix_matrix)/(pix_matrix.max())
pix_matrix = pix_matrix**.55
vertical_pixels, horizontal_pixels = pix_matrix.shape
Ny = vertical_pixels
Nx = horizontal_pixels
axs[0,0].imshow(pix_matrix)
axs[0,0].set_title("Normalized Image")

#Set a noise floor
pix_norm = (pix_matrix-pix_matrix.std())/(pix_matrix.max()-pix_matrix.std())
pix_norm[pix_norm<0]=0

#Smooth the data
fake_smooth = sp.ndimage.gaussian_filter(pix_norm,1,truncate=np.sqrt(Nx*Ny)/2)
fake_smooth = (fake_smooth-fake_smooth.min())/(fake_smooth.max()-fake_smooth.min())
axs[0,1].imshow(fake_smooth)
axs[0,1].set_title("Remove Noise Floor then Smooth")

# Set values below a threshold to zero
fake_local_norm = fake_smooth/sp.ndimage.maximum_filter(fake_smooth,10)
fake_local_norm[fake_smooth<2*pix_norm.std()]=0
fake_local_norm = sp.ndimage.gaussian_filter(fake_local_norm,1)
axs[1,0].imshow(fake_local_norm)
contours = axs[1,0].contour(fake_local_norm, levels=[.5], colors="white", extent=(0, Nx, 0, Ny))
axs[1,0].set_title("Plot Contours at Half Maxima Values")

#Set the aspect ratio of the plots
for axi in axs:
    for ax in axi:
        ax.set_aspect(Ny*1.0/Nx)

#This grabs the coutours from the first value and finds the center point
xs = []
ys = []
dxs = []
dys = []
drs = []

for seg in contours.allsegs[0]:
    center = np.mean(seg,axis=0)
    dcenter = np.std(seg,axis=0)
    dr = np.sqrt(dcenter[0]**2+dcenter[1]**2)
    #Don't count contours arrount a single pixel
    if dr>1:
        xs.append(center[0])
        dxs.append(dcenter[0])
        ys.append(center[1])
        dys.append(dcenter[1])
        drs.append(dr)
locations = pd.DataFrame(data={"x":xs,"dx":dxs,"y":ys,"dy":dys,"dr":drs})

axs[1,1].imshow(pix_matrix)
axs[1,1].errorbar(x=locations["x"],
                  y=locations["y"],
                  xerr=locations["dx"],
                  yerr=locations["dy"],fmt="r",ls="none")
axs[1,1].set_title("Plot Star Locations")
axs[1,1].legend(loc='center left', bbox_to_anchor=(1, 0.5),facecolor="grey")

#Calcualte an anisotropy value
#anything over 0.25 might be two overlaping objects
# locations["anisotropy"] = np.abs(locations["dy"]-locations['dx'])/(locations["dy"]+locations['dx'])

print(locations)
fig.show()
fig.savefig("test.png")