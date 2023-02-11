'''

'''

import numpy as np

from scipy.ndimage  import gaussian_filter, grey_dilation, convolve
from tqdm           import tqdm_notebook as tqdm

from .read import load_image, load_movie_frames
from .utils import get_frames
from .plots import plots


class Tracking():
    '''
    Main class to track individual particles from 2D images

    Attributes
    ----------
        - params : dict
            Dictionary including the following keys and filter_sizes ->   directory : str
                                                                    filename : str
                                                                    media : str
                                                                    mode : str
                                                                    format : str
                                                                    nframes : int
                                                                    diameter : int
                                                                    filter_size : int
                                                                    bgavg : int
                                                                    minbright : int
                                                                    masscut : int
                                                                    cutoff : int
                                                                    initial_frame : int
                                                                    single_frame : int
                                                                    plot : bool


    Methods
    -------
        - get_convolve_image(img:array,mask:array)
            produces convoluttion between image and mask
        
        - get_dilated_image(img:array)
            produces dilated images

        - get_circular_mask(filter_size:int)
            generates as circular mask with size as funtion of filter_size filter_size

        - get_image_subregion(centre:float, radius:float, img:np.ndarray)
            adds and removes a radius in pixels to the image
    '''
    def __init__(self,params):
        self.directory      = params.get('directory')
        self.filename       = params.get('filename')
        self.media          = params.get('media')
        self.mode           = params.get('mode')
        self.format         = params.get('format')
        self.nframes        = params.get('nframes')
        self.diameter       = params.get('diameter')
        self.filter_size    = params.get('filter_size')
        self.bgavg          = params.get('bgavg')
        self.minbright      = params.get('minbright')
        self.masscut        = params.get('masscut')
        self.cutoff         = params.get('cutoff')
        self.initial_frame  = params.get('initial_frame')
        self.frame_rate     = params.get('frame_rate')
        self.single_frame   = params.get('single_frame')
        self.plot           = params.get('plot')

    def get_convolve_image(self, img:np.ndarray, mask:np.ndarray):
        return convolve(img, mask)

    def get_dilated_image(self, img:np.ndarray):
        return grey_dilation(img, size=(self.diameter,self.diameter))

    def get_circular_mask(self, filter_size:int):
        kernel  = np.zeros((2 * filter_size + 1, 2 * filter_size + 1))
        y,x     = np.ogrid[-filter_size : filter_size + 1, -filter_size : filter_size + 1]
        mask    = x**2 + y**2 <= filter_size**2
        kernel[mask] = 1
        return kernel

    def get_image_subregion(self, centre:float, radius:float, img:np.ndarray):
        '''
        adds and removes a radius in pixels to the image
        '''
        y           = centre[0]
        x           = centre[1]
        xlo, xhi    = x - radius, x + radius
        ylo, yhi    = y - radius, y + radius
        img_sub     = img[int(np.ceil(ylo)):int(np.ceil(yhi)),int(np.ceil(xlo)):int(np.ceil(xhi))]
        return img_sub

    def get_lowpass_filter(self, img:np.ndarray):
        '''
        applies convolution between original image and a generated circular mask
        '''
        img_0           = img
        img_1           = img_0 - np.mean(img_0)
        # img_1[img_1 < 0] = 0
        img_1           /= img_1.max()
        circular_mask   = self.get_circular_mask(self.filter_size)**2.
        circular_mask   /= np.sum(circular_mask)
        lowpass_img     = self.get_convolve_image(img_1, circular_mask)
        backgceil       = np.ones((self.bgavg, self.bgavg))
        backgceil       /= self.bgavg**2.
        bg_img          = self.get_convolve_image(img_1, backgceil)
        img_2           = lowpass_img - bg_img
        img_2[img_2 < 0] = 0
        img_2 /= img_2.max()
        return img_2

    def cut_edge(self, matrix:np.ndarray, img_shape:tuple):
        '''
        cuts N edge pixels from the original image
        '''
        height  = img_shape[0]
        width   = img_shape[1] 
        return matrix[np.where((matrix[:,0] > self.cutoff) & (matrix[:,0] < (width - self.cutoff)) & (matrix[:,1] > self.cutoff) & (matrix[:,1] < (height - self.cutoff)))]

    def filter_overlaps(self,coords:np.ndarray):
        '''
        removes different particles with same coordinates. Useful for linking trajectories later
        '''
        r = np.sqrt(coords[:,0]**2 + coords[:,1]**2)
        _,uniques = np.unique(r,return_index = True)
        return coords[uniques]

    def get_particles(self, img:np.ndarray):
        '''
        identifies particles based on image dilation and brightness threshold

        :param img  : np.ndarray. Low passe filtered image
        :return     : np.ndarray. Particle coordinates
        '''
        ny          = img.shape[0]
        nx          = img.shape[1]

        border_img  = np.zeros((ny + 2*self.diameter, nx + 2*self.diameter))
        border_img[self.diameter:self.diameter+ny, self.diameter:self.diameter+nx] = img
        
        radius          = self.diameter / 2.0
        range           = (self.diameter - 1) / 2
        circular_mask   = self.get_circular_mask(int(radius))
        # dilated_img = self.dilation(border_img, circmask)
        dilated_img     = self.get_dilated_image(border_img)

        #Now get initial coordinates. For now this are int px values
        r = ((dilated_img == border_img) & (border_img > self.minbright))
        initial_coords = np.vstack(np.where(r)).T

        #Now get particles after separation
        initial_x_coords = initial_coords[:,1]
        initial_y_coords = initial_coords[:,0]
        xlo, xhi = initial_x_coords - radius, initial_x_coords + radius
        ylo, yhi = initial_y_coords - radius, initial_y_coords + radius
        bimgr = border_img * r

        for i in np.arange(initial_coords.shape[0]):
            img_sub = self.get_image_subregion(initial_coords[i], radius, bimgr)
            img_sub = img_sub * circular_mask 
            maxima = np.where(img_sub == img_sub.max())
            if img_sub.max() != 0:
                if any((img_sub[0].shape == maxima[0]) & (img_sub[1].shape == maxima[1])):
                    if ((maxima[0] - (range + 1))**2 + (maxima[1] - (range + 1))**2) != 0:
                        bimgr[initial_y_coords[i], initial_x_coords[i]] = 0
                #This part needs sorting
                # else:
                #     if all(((maxima[0] - (rn+1)**2) + (maxima[1] - (rn+1))**2) !=0):
                #         bimgr[initial_y_coords[i], initial_x_coords[i]] = 0
                #     else:
                #         avgmaxx = np.ceil(np.mean(maxima[1]))
                #         avgmaxy = np.ceil(np.mean(maxima[0]))
                #         bimgr[int(np.ceil(ylo[0])):int(np.ceil(yhi[0])),int(np.ceil(xlo[0])):int(np.ceil(xhi[0]))] = 0
                #         bimgr[int(np.ceil(ylo[i])) + int(avgmaxy), int(np.ceil(xlo[i])) + int(avgmaxx)] = 1
        asep_coords     = np.vstack(np.where(bimgr !=0)).T
        asep_x_coords   = asep_coords[:,1]
        asep_y_coords   = asep_coords[:,0]

        #Now particles after masscut
        nxlo, nxhi = asep_x_coords - radius, asep_x_coords + radius
        nylo, nyhi = asep_y_coords - radius, asep_y_coords + radius
        mass = np.zeros((asep_coords.shape[0], 1))

        for j in np.arange(asep_coords.shape[0]):
            mass[j,0] = np.sum(border_img[int(np.ceil(nylo[j])):int(np.ceil(nyhi[j])),int(np.ceil(nxlo[j])):int(np.ceil(nxhi[j]))] * circular_mask)

        masstrue            = np.where(mass > self.masscut)
        masstrue            = masstrue[0]
        amasscut_x_coords   = asep_x_coords[masstrue]
        amasscut_y_coords   = asep_y_coords[masstrue]
        mass                = mass[masstrue]
        am_xlo              = xlo[masstrue]
        am_xhi              = xhi[masstrue]
        am_ylo              = ylo[masstrue]
        am_yhi              = yhi[masstrue]

        #Now particles after eccentricity
        x_centers           = np.zeros((amasscut_x_coords.shape[0],1))
        y_centers           = np.zeros((amasscut_y_coords.shape[0],1))

        seq                 = np.arange(-radius, radius,1)
        xmask               = np.zeros((self.diameter, self.diameter))
        xmask[:]            = seq
        ymask               = xmask.T
        xmask               = xmask * circular_mask
        ymask               = ymask * circular_mask

        for k in np.arange(amasscut_x_coords.shape[0]):
            x_centers[k,0] = np.sum(border_img[int(np.ceil(am_ylo[k])):int(np.ceil(am_yhi[k])),int(np.ceil(am_xlo[k])):int(np.ceil(am_xhi[k]))]*xmask)
            y_centers[k,0] = np.sum(border_img[int(np.ceil(am_ylo[k])):int(np.ceil(am_yhi[k])),int(np.ceil(am_xlo[k])):int(np.ceil(am_xhi[k]))]*ymask)

        x_centers           = x_centers / mass
        y_centers           = y_centers / mass
        aecc_x_coords       = amasscut_x_coords.reshape(amasscut_x_coords.shape[0],1) + x_centers
        aecc_y_coords       = amasscut_y_coords.reshape(amasscut_y_coords.shape[0],1) + y_centers

        #Create coords matrix
        coords              = np.zeros((aecc_x_coords.shape[0],2))
        coords[:,0]         = aecc_x_coords[:,0] - self.diameter
        coords[:,1]         = aecc_y_coords[:,0] - self.diameter
        coords[:,1]         = -coords[:,1] + ny #invert y-coords
        return coords

    def tracking(self):
        '''
        Main function. Reads IMAGES or FRAMES_FROM_MOVIE in MULTI or SINGLE mode, which means analysing one or many images.

        Returns dictionary containing all found coordinates per frame
        '''

        frames = {}

        if self.mode == 'None':
            raise ValueError('Select single or multi tracking mode')

        if self.mode == 'single':
            if self.media == 'Images':
                img = load_image(self.directory, self.filename, self.single_frame, self.format)
            elif self.media == 'Movie':
                img = load_movie_frames(self.directory,self.filename,self.single_frame,self.format)

            lp_img  = self.get_lowpass_filter(img)
            ny      = lp_img.shape[0]

            coords          = self.get_particles(lp_img)
            cut_coords      = self.cut_edge(coords,img.shape)
            frame_number    = np.zeros((cut_coords.shape[0],1))
            frame_number[:] = self.single_frame
            coords_upd      = np.hstack((cut_coords, frame_number))
            filtered_coords = self.filter_overlaps(coords_upd)
            frames[self.single_frame] = filtered_coords

        elif self.mode == 'multi':
            fr = np.arange(self.initial_frame,self.nframes + self.initial_frame,1)
            pbar =  tqdm(enumerate(fr))
            for i,nf in pbar:
                pbar.set_description('Loading frame: {} of {}'.format(i + 1, self.nframes))

                if self.media == 'Images':
                    img = load_image(self.directory, self.filename,nf,self.format) #original image as np.array
                elif self.media == 'Movie':
                    img = load_movie_frames(self.directory,self.filename,nf,self.format)

                lp_img  = self.get_lowpass_filter(img)
                ny      = lp_img.shape[0]
                coords  = self.get_particles(lp_img)

                # Now apply edge_cutoff
                cut_coords = self.cut_edge(coords,img.shape)

                #Add frame number to coords matrix
                frame_number    = np.zeros((cut_coords.shape[0],1))
                frame_number[:] = nf
                coords_upd      = np.hstack((cut_coords, frame_number))

                filtered_coords = self.filter_overlaps(coords_upd)
                frames[nf]      = filtered_coords

        if self.plot== True:
            plots.over_plot(self.media,self.mode,self.directory,self.nframes,self.filename,self.single_frame,self.format,frames,ny,self.frame_rate)

        return frames