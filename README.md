# Forklens
### A convolutional neural network (CNN) to measure weak gravitational lensing signal, namely cosmic shear.

------------------------------------------------------------------------------------------

![d1a4058a-979b-4963-8aa3-11e04e9c8537](https://user-images.githubusercontent.com/31132161/208124496-0a75bac7-c328-46b6-8d97-734868888a0a.jpg)

**(Image by OpenAI)**

----------------------------------------------------------------------------------------

For more details, please refer to ... and also to cite it if use.

This project is originally motivated to measure galaxy shapes (shear) and in the meantime correct the smearing of the point spread function (PSF, an effect from either/both the atmosphere and optical instrument). The code contains a custom CNN achitecture which has two input branches, fed with the observed galaxy image and PSF image, predicting several features of the galaxy (shapes, magnitude, size, etc.). Simulation in the code is built directly upon Galsim.


## To install

>> cd forklens
>> 
>> python setup.py install

Check some example tests in ./examples.


## Requirements
Python==3.9.7, Pytorch==1.11.0+cu113, Galsim==2.3.4
