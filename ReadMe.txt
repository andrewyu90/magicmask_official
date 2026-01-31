This is the source code of MagicMask Demo.

1. Install requirement.txt file
2. Download encoder's trained parameters from 'https://drive.google.com/file/d/1qc4s6eRQPluma72WFibUnw74GPMAYRtY/view?usp=sharing'
3. Download pretrained_model from 'https://drive.google.com/file/d/14ti8lMXF9AYVS8CyTaFDyU6tpsK-qWjV/view?usp=sharing'
2. Run jupyer notebook for 'test.ipynb'

Sourc image size is 112 x 112.

For custom inference, 

you need to reguialise source and target images as follows

For source image.
 - Size: 112x112x3 (RGB)
 - Value  (x/127.5)-1.0, where x is image.

For target image
 - size: 128x128x3 (RGB)
 - value x/255.0