import galsim
import numpy as np
from numpy.random import Generator, PCG64
from astropy.io import fits

import sys
# import os
# sys.path.append(os.path.abspath("../configs"))
# from configs import config
import config


def AddBias(img, bias_level=500, nsecy=2, nsecx=8, seed=202102):
    # Generate Bias and its non-uniformity, and add the 16 bias values to the GS-Image
    rg = Generator(PCG64(int(seed)))
    Random16 = (rg.random(nsecy*nsecx)-0.5)*20
    if int(bias_level)==0:
        BiasLevel = np.zeros((nsecy,nsecx))
    elif bias_level>0:
        BiasLevel = Random16.reshape((nsecy,nsecx)) + bias_level

    arrshape = img.array.shape
    secsize_x = int(arrshape[1]/nsecx)
    secsize_y = int(arrshape[0]/nsecy)
    for rowi in range(nsecy):
        for coli in range(nsecx):
            img.array[rowi*secsize_y:(rowi+1)*secsize_y,coli*secsize_x:(coli+1)*secsize_x] += BiasLevel[rowi,coli]
    return img

def ApplyGain(img, gain=1, nsecy = 2, nsecx=8, seed=202102, logger=None):
    # Generate Gain non-uniformity, and multipy the different factors (mean~1 with sigma~1%) to the GS-Image
    rg = Generator(PCG64(int(seed)))
    Random16 = (rg.random(nsecy*nsecx)-0.5)*0.04+1   # sigma~1%
    Gain16 = Random16.reshape((nsecy,nsecx))/gain
    
    arrshape = img.array.shape
    secsize_x = int(arrshape[1]/nsecx)
    secsize_y = int(arrshape[0]/nsecy)
    for rowi in range(nsecy):
        for coli in range(nsecx):
            img.array[rowi*secsize_y:(rowi+1)*secsize_y,coli*secsize_x:(coli+1)*secsize_x] *= Gain16[rowi,coli]
    return img


#set noiseModel for CSST-image
def addDetNoise(img):

    seed=20221223+np.random.randint(10000)
    rng_poisson = galsim.BaseDeviate(seed)

    if config.simulation['sky_background'] != None:
        sky_map = config.simulation['sky_background']
        ###update-START-20221106
        sky_map = sky_map * np.ones_like(img.array)
        sky_map = galsim.Image(array=sky_map)
        poisson_noise = galsim.PoissonNoise(rng_poisson, sky_level=0.)
        sky_map.addNoise(poisson_noise)
        ###update-END
        img += sky_map
        del sky_map

    if config.simulation['dark_noise'] != None:
        darkNoise = galsim.DeviateNoise(galsim.PoissonDeviate(rng_poisson, 
                                                              config.simulation['dark_noise']))
        img.addNoise(darkNoise)

    if config.simulation['read_noise'] != None:
        seed = seed+6
        rng_readout = galsim.BaseDeviate(seed)
        readNoise = galsim.GaussianNoise(rng=rng_readout, 
                                         sigma=config.simulation['read_noise'])
        img.addNoise(readNoise)
        
    if config.simulation['bias_level'] != None:
        img = AddBias(img, bias_level=config.simulation['bias_level'], nsecy=1, nsecx=1, seed=seed+5)

    if config.simulation['gain'] != None:
        img = ApplyGain(img, gain=config.simulation['gain'], nsecy=1, nsecx=1, seed=seed+7)

    return img

#get nphotons from given mag in i band for CSST
def mag2photon(mag):
    # convert the AB magnitude to photons
    # mag = -2.5*log10(fnu) - 48.6
    # fnu is in unit of erg s-1 cm-2 Hz-1
    
    magLim= 25.9
    cband = 7653.2
    wband = 1588.1
    iphotonEnergy = 2.5956248184155225e-12
    vc = 2.99792458e+18 # speed of light: A/s
    exp_time = 150
    
    vc = 2.99792458e+18 # speed of light: A/s
    h_Planck = 6.626196e-27 # Planck constant: erg s
    
    lowband = cband - 0.5*wband
    upband  = cband + 0.5*wband

    iflux = 10**(-0.4*(mag+48.6)) #mag2flux(mag) #erg/s/cm^2/Hz
    nn = iflux/iphotonEnergy
    nn = nn * vc*(1.0/lowband - 1.0/upband)
    nObsPhoton = nn   #photons/cm2/s

    nObsPhoton = nObsPhoton * 1e4 * np.pi * exp_time
    return nObsPhoton / 2

#define galaxy in galsimObj
def get_gal(gal_param=None, shear=None):
    #thetaR   = gal_param["theta"]
    # bfrac    = gal_param["bfrac"]
    hlr_disk = gal_param["hlr_disk"]
    # hlr_bulge= gal_param["hlr_bulge"]

    # Extract ellipticity components
    e_disk = galsim.Shear(g1=gal_param["e1"], g2=gal_param["e2"])
    # e_bulge = galsim.Shear(g=gal_param["ell_bulge"], beta=thetaR*galsim.degrees)
    # e_total = galsim.Shear(g=gal_param["ell_tot"], beta=thetaR*galsim.degrees)
    e1_disk, e2_disk  = e_disk.g1, e_disk.g2
    # e1_bulge, e2_bulge= e_bulge.g1, e_bulge.g2
    # e1_total, e2_total= e_total.g1, e_total.g2

    disk = galsim.Sersic(n=1.0, half_light_radius=hlr_disk, flux=1.0)
    disk_shape = galsim.Shear(g1=e1_disk, g2=e2_disk)
    disk = disk.shear(disk_shape)
    # bulge = galsim.Sersic(n=4.0, half_light_radius=hlr_bulge, flux=1.0)
    # bulge_shape = galsim.Shear(g1=e1_bulge, g2=e2_bulge)
    # bulge = bulge.shear(bulge_shape)

    # gal = bfrac * bulge + (1.0 - bfrac) * disk
    gal = disk

    tmag= gal_param['mag_i']
    nn  = mag2photon(tmag)
    gal = gal.withFlux(nn)

    if not (shear is None):
        # gal_shear = galsim.Shear(g1=shear[0], g2=shear[1])
        gal = gal.shear(g1=shear[0], g2=shear[1])
    return gal


def get_psf(psf_pars):
    
    # psf = galsim.Airy(
    #     psf_pars[3]/1.025,
    #     flux=1.0,
    #     obscuration=0.,
    #     scale_unit=galsim.arcsec
    # ).shear(g1=psf_pars[0], g2=psf_pars[1])
    
    # One example of CSST-like PSF
    with fits.open('../data/csst_psf_9999.fits') as f:
        psf_im = f[0].data
        
    img = galsim.ImageF(psf_im, scale=config.simulation['pixel_size'])
    psf = galsim.InterpolatedImage(img)

    return psf


def get_sim(gal_param, psf_param, shear=None):
    
    scale = config.simulation['pixel_size']
    gal_stamp = config.simulation['galaxy_stamp_size']
    psf_stamp = config.simulation['psf_stamp_size']
    
    gal = get_gal(gal_param=gal_param, shear=shear)
    psf = get_psf(psf_param)
    gal = galsim.Convolve(psf, gal)

    phot= gal.drawImage(method='phot', poisson_flux=False, save_photons=True, scale=scale).photons
    phot.x += gal_stamp/2
    phot.y += gal_stamp/2

    stamp = galsim.ImageF(ncol=gal_stamp, nrow=gal_stamp, scale=scale)
    stamp.setCenter(gal_stamp/2, gal_stamp/2)
    sensor = galsim.Sensor()
    sensor.accumulate(phot, stamp)
    
    clean_gal = stamp.array.copy()
    stamp = addDetNoise(stamp)
        
    gal_im = stamp.array
    psf_im = psf.drawImage(nx=psf_stamp,ny=psf_stamp,scale=scale).array
    label = np.array([gal_param['e1'],
                      gal_param['e2'],
                      gal_param['hlr_disk'],
                      gal_param['mag_i'],
                      # psf_param['randint'],
    ])
    return gal_im, clean_gal, psf_im, label