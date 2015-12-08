#############################################################
#            Created by Yinan Xu, 12/06/2015                #
#            Copyright @ Yinan Xu                           #
#############################################################

import Image
from utils import tile_raster_images
import load_CIFAR as LF


datasets = LF.load_cifar()
train_set_x, train_set_y = datasets[0]
xx = train_set_x.get_value()[0:100]

image = Image.fromarray(
    tile_raster_images(
        X=xx,
        img_shape=(32, 32),
        tile_shape=(10, 10),
        tile_spacing=(1, 1)
    )
)
image.save('first_100_images.png')
print "save images successful"





