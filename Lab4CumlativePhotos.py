import pandas as pd
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from datetime import datetime

plt.rcParams['figure.figsize'] = [9, 9]

fit_files = ['Jupiter/Jupiter_Light11-19-1.fits','Jupiter/Jupiter_Light11-19-2.fits','Jupiter/Jupiter_Light11-19-3.fits','Jupiter/Jupiter_Light11-20-1.fits','Jupiter/Jupiter_Light11-20-2.fits','Jupiter/Jupiter_Light11-20-3.fits','Jupiter/Jupiter_Light11-20-4.fits','Jupiter/Jupiter_Light11-20-5.fits','Jupiter/Jupiter_Light11-20-6.fits','Jupiter/Jupiter_Light11-20-7.fits','Jupiter/Jupiter_Light11-20-8.fits','Jupiter/Jupiter_Light11-20-9.fits','Jupiter/Jupiter_Light11-21-1.fits','Jupiter/Jupiter_Light11-21-2.fits','Jupiter/Jupiter_Light11-21-3.fits','Jupiter/Jupiter_Light11-21-4.fits','Jupiter/Jupiter_Light11-21-8.fits']
#Files with more light exposure
high_light = ['19-2','19-3','21-1','21-2','21-3','21-4','21-8']
gamma = .22

def read_fits(file_path):
    with fits.open(file_path) as hdul:
        data = hdul[0].data
    return data

def create_pix_matrix(fits_files):
    pix_matrices = {'File Path':'Image Matrix'}
    for fits_file in fits_files:
        data = np.array(read_fits(fits_file))
        pix_matrices.update({fits_file : np.squeeze(data)})
    return pix_matrices

def get_observation_date(file_path):
    with fits.open(file_path) as hdul:
        header = hdul[0].header
        date_obs_str = header.get('DATE-OBS', None)
        datetime_obj = datetime.fromisoformat(date_obs_str)
        reference_date = datetime(1970, 1, 1)
        seconds_since_epoch = (datetime_obj - reference_date).total_seconds()
        return seconds_since_epoch

# def get_dJ(x,y,dr):


image_matrices = create_pix_matrix(fit_files)
print(image_matrices)
file_locations = {'File Path':'Locations'}

for file in fit_files:
    #Get the pixel matrix for each image
    img_matrix = image_matrices[file]

    #Normalize pixel values and get size of image
    img_matrix = (img_matrix)/(img_matrix.max())
    for substring in high_light:
        if substring in file:
            img_matrix = img_matrix**.6
            break
    else:
        img_matrix = img_matrix**gamma
    vertical_pixels, horizontal_pixels = img_matrix.shape
    Ny = vertical_pixels
    Nx = horizontal_pixels

    #setup the plotting axes
    fig, axs = plt.subplots(nrows=2, ncols=2)

    axs[0,0].imshow(img_matrix)
    axs[0,0].set_title("Normalized Image")

    #Set a noise floor
    pix_norm = (img_matrix-img_matrix.std())/(img_matrix.max()-img_matrix.std())
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
    axs[1,1].imshow(img_matrix)
    axs[1,1].errorbar(x=locations["x"],
                  y=locations["y"],
                  xerr=locations["dx"],
                  yerr=locations["dy"],fmt="r",ls="none")
    axs[1,1].set_title("Plot Star Locations")
    axs[1,1].legend(loc='center left', bbox_to_anchor=(1, 0.5),facecolor="grey")
    file_locations.update({file:locations})
    fig.savefig("JupiterPlots/" + file.replace(".fits", ".png"))

for file in fit_files:
    df = file_locations[file]
    xj = df[df['dr'] == df['dr'].max()]['x']
    yj = df[df['dr'] == df['dr'].max()]['y']
    df['dxj'] = pd.to_numeric(df['x']) - float(xj)
    df['dyj'] = pd.to_numeric(df['y']) - float(yj)
    df['time'] = get_observation_date(file)
    df['file_name'] = file
    try:
        alldf = pd.concat([alldf, df], ignore_index=True)
    except:
        alldf = pd.DataFrame(df)
print(alldf)

fig, axs = plt.subplots(nrows=1, ncols=1)

filtered_df = alldf[(alldf['dxj'] != 0)]

plt.scatter(pd.to_numeric(filtered_df['time']), filtered_df['dxj'],
            c=pd.to_numeric(filtered_df['dr']), cmap='viridis', label='DR values', alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Distance from Jupiter in the x-plane (pixels)')
plt.title('dxj vs time')
cbar = plt.colorbar()
cbar.set_label('Moon Radius (pixels)')
plt.show()

plt.scatter(pd.to_numeric(filtered_df['time']), filtered_df['dyj'],
            c=pd.to_numeric(filtered_df['dr']), cmap='viridis', label='DR values', alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Distance from Jupiter in the y-plane (pixels)')
plt.title('dyj vs time')
cbar = plt.colorbar()
cbar.set_label('Moon Radius (pixels)')
plt.show()

filtered_df['dj'] = np.sqrt((filtered_df['dxj'])**2 + (filtered_df['dyj'])**2)

plt.scatter(pd.to_numeric(filtered_df['time']), (filtered_df['dj'])/((np.abs(filtered_df['dxj']))/filtered_df['dxj']),
            c=pd.to_numeric(filtered_df['dr']), cmap='viridis', label='DR values', alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Distance from Jupiter in the 2D-plane (pixels)')
plt.title('dj vs time')
cbar = plt.colorbar()
cbar.set_label('Moon Radius (pixels)')
plt.show()

#11-21(7,6,5)