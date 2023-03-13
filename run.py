#!/usr/bin/env python3

import subprocess
import argparse
import PIL.Image
import math


# parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("scalefactor", type=int)
ap.add_argument("psfsigma", type=float)
ap.add_argument("action", type=str)
args = ap.parse_args()


img = PIL.Image.open('input_0.png')
(sizeX, sizeY) = img.size
p = {}

if args.action == "Interpolate":
    # In this run mode, the interpolation is performed directly on the
    # selected image, and the estimated contours are also shown.
    
    # If the image dimensions are small, zoom the displayed results.
    # This value is always at least 1.
    displayzoom = int(math.ceil(400.0/(args.scalefactor*max(sizeX, sizeY))))
    # Check that interpolated image dimensions are not too large
    cropsize = (min(sizeX, int(400/args.scalefactor)),
        min(sizeY, int(400/args.scalefactor)))
    
    if (sizeX, sizeY) != cropsize:
        (x0, y0) = (int(math.floor((sizeX - cropsize[0])/2)),
            int(math.floor((sizeY - cropsize[1])/2)))
        imgcrop = img.crop((x0, y0, x0 + cropsize[0], y0 + cropsize[1]))
        imgcrop.save('input_0.png')
    
    p = {
        # Perform the actual contour stencil interpolation
        'interp' : 
            subprocess.run(['tdinterp', '-x', str(args.scalefactor), '-p', str(args.psfsigma), 'input_0.png', 'interpolated.png']),
            
        # Interpolate with Fourier
        'finterp' : 
            subprocess.run(['tdinterp', '-N0', '-x', str(args.scalefactor), '-p', str(args.psfsigma), 'input_0.png', 'fourier.png']),
            
        # For display, create a nearest neighbor zoomed version of the
        # input. nninterp does nearest neighbor interpolation on 
        # precisely the same grid so that displayed images are aligned.
        'inputzoom' : 
            subprocess.run(['nninterp', '-g', 'centered', '-x', str(args.scalefactor*displayzoom), 'input_0.png', 'input_0_zoom.png'])
        }
    
    if displayzoom > 1:
        subprocess.run(['nninterp', '-g', 'centered', '-x', str(displayzoom), 'interpolated.png', 'interpolated_zoom.png'])
        subprocess.run(['nninterp', '-g', 'centered', '-x', str(displayzoom), 'fourier.png', 'fourier_zoom.png'])

else:
    #write Coarsen=True in algo_info.txt
    with open('algo_info.txt', 'w') as file:
        file.write("Coarsen=1")

    # In this run mode, the selected image is coarsened, interpolated
    # and compared with the original.
    
    # Check that interpolated image dimensions are not too large
    cropsize = (min(sizeX, 400), min(sizeY, 400))
    
    if (sizeX, sizeY) != cropsize:
        (x0, y0) = (int(math.floor((sizeX - cropsize[0])/2)),
            int(math.floor((sizeY - cropsize[1])/2)))
        imgcrop = img.crop((x0, y0, x0 + cropsize[0], y0 + cropsize[1]))
        imgcrop.save('input_0.png')
        (sizeX, sizeY) = cropsize
    
    # If the image dimensions are small, zoom the displayed results.
    # This value is always at least 1.
    displayzoom = int(math.ceil(350.0/max(sizeX, sizeY)))
    
    # Coarsen the image
    p['coarsened'] = subprocess.run(['imcoarsen', '-g', 'topleft', '-x', str(args.scalefactor), '-p', str(args.psfsigma), 'input_0.png', 'coarsened.png'])
    
    if displayzoom > 1:
        p['exactzoom'] = subprocess.run(['nninterp', '-g', 'centered', '-x', str(displayzoom), 'input_0.png', 'input_0_zoom.png'])
    
    # Perform the actual interpolation
    p['interpolated'] = subprocess.run(['tdinterp', '-x', str(args.scalefactor), '-p', str(args.psfsigma), 'coarsened.png', 'interpolated.png'])
    
    # Interpolate with Fourier
    p['fourier'] = subprocess.run(['tdinterp', '-N0', '-x', str(args.scalefactor), '-p', str(args.psfsigma), 'coarsened.png', 'fourier.png'])

    # For display, create a nearest neighbor zoomed version of the
    # coarsened image.  nninterp does nearest neighbor interpolation 
    # on precisely the same grid as cwinterp so that displayed images
    # are aligned.
    p['coarsened_zoom'] = subprocess.run(['nninterp', '-g', 'topleft', '-x', str(args.scalefactor), 'coarsened.png', 'coarsened_zoom.png'])
                
    # Because of rounding, the interpolated image dimensions might be 
    # slightly larger than the original image.  For example, if the 
    # input is 100x100 and the scale factor is 3, then the coarsened 
    # image has size 34x34, and the interpolation has size 102x102.
    # The following crops the results if necessary.
    for f in ['coarsened_zoom', 'interpolated', 'fourier']:
        img = PIL.Image.open(f + '.png')
        
        if (sizeX, sizeY) != img.size:
            imgcrop = img.crop((0, 0, sizeX, sizeY))
            imgcrop.save(f + '.png')
                
    # Generate difference image
    p['difference'] = subprocess.run(['imdiff', 'input_0.png', 'interpolated.png', 'difference.png'])
    p['fdifference'] = subprocess.run(['imdiff', 'input_0.png', 'fourier.png', 'fdifference.png'])
    # Compute maximum difference, PSNR, and MSSIM
    p['metrics'] = subprocess.run(['imdiff', 'input_0.png', 'interpolated.png'])
    
    if displayzoom > 1:
        p['coarsened_zoom2'] = subprocess.run(['nninterp', '-g', 'centered', '-x', str(displayzoom), 'coarsened_zoom.png', 'coarsened_zoom.png'])
        
        for f in ['interpolated', 'fourier', 'difference', 'fdifference']:
            p[f + '_zoom'] = subprocess.run(['nninterp', '-g', 'centered', '-x', str(displayzoom), f + '.png', f + '_zoom.png'])
