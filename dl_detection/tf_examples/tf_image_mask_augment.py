import tensorflow as tf
import tensorflow_addons as tfa
import math


''' The function is modified from code https://becominghuman.ai/data-augmentation-on-gpu-in-tensorflow-13d14ecf2b19 '''
''' 04/16/2018 add support for 3D image, xy flip/rotation/crop, which is OK for cell shapes'''
''' 09/28/2018 add augmentation for (2D) outline '''
''' 08/13/2019 augment for both image and mask'''

def augment(images, masks,
            resize=None, # (width, height) tuple or None
            crop_shape=None,  # crop image to small size (width, height)
            horizontal_flip=False,
            vertical_flip=False,
            rotate=0, # Maximum rotation angle in degrees
            crop_probability=0, # How often we do crops
            crop_min_percent=0.6, # Minimum linear dimension of a crop
            crop_max_percent=1.,  # Maximum linear dimension of a crop
            mixup=0):  # Mixup coeffecient, see https://arxiv.org/abs/1710.09412.pdf

    # My experiments showed that casting on GPU improves training performance
    if type(images) != tf.float32:
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        # images = tf.subtract(images, 0.5)
        # images = tf.multiply(images, 2.0)
        masks = tf.image.convert_image_dtype(masks, dtype=tf.float32)

    if resize is not None:
        images = tf.image.resize_bilinear(images, resize)
        masks = tf.image.resize_nearest_neighbor(masks, resize)

    if crop_shape is not None:
        shp = tf.shape(images)
        print(shp)
        batch_size, height, width = shp[0], shp[1], shp[2]
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)

        # hrg, wrg = crop_shape[0], crop_shape[1]
        normal_h = crop_shape[0] / height
        normal_w = crop_shape[1] / width

        h_offset = tf.random.uniform([batch_size, 1], 0, 1 - normal_h)
        w_offset = tf.random.uniform([batch_size, 1], 0, 1 - normal_w)

        boxes = tf.concat([h_offset, w_offset, h_offset, w_offset], 1)
        boxes = boxes + tf.convert_to_tensor([0, 0, normal_h, normal_w], dtype=tf.float32)

        boxes_inds = tf.range(batch_size)

        images = tf.image.crop_and_resize(images, boxes=boxes, box_indices=boxes_inds, crop_size=crop_shape, method='bilinear')
        masks = tf.image.crop_and_resize(masks, boxes=boxes, box_indices=boxes_inds, crop_size=crop_shape, method='nearest')

    shp = tf.shape(images)
    batch_size, height, width = shp[0], shp[1], shp[2]
    width = tf.cast(width, tf.float32)
    height = tf.cast(height, tf.float32)
    # print(batch_size)

    # The list of affine transformations that our image will go under.
    # Every element is Nx8 tensor, where N is a batch size.
    transforms = []
    identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
    if horizontal_flip:
        coin = tf.less(tf.compat.v1.random_uniform([batch_size], 0, 1.0), 0.5)
        flip_transform = tf.convert_to_tensor(
          [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
        transforms.append(tf.where(tf.expand_dims(coin, 1),
                                 tf.expand_dims(flip_transform, 0), tf.expand_dims(identity, 0)))

    if vertical_flip:
        coin = tf.less(tf.compat.v1.random_uniform([batch_size], 0, 1.0), 0.5)
        flip_transform = tf.convert_to_tensor(
          [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
        transforms.append(tf.where(tf.expand_dims(coin, 1),
                                 tf.expand_dims(flip_transform, 0), tf.expand_dims(identity, 0)))

    if rotate > 0:
        angle_rad = rotate / 180 * math.pi
        angles = tf.compat.v1.random_uniform([batch_size], -angle_rad, angle_rad)
        transforms.append(tfa.image.transform_ops.angles_to_projective_transforms(angles, height, width))

    if crop_probability > 0:
        crop_pct = tf.compat.v1.random_uniform([batch_size], crop_min_percent, crop_max_percent)
        left = tf.compat.v1.random_uniform([batch_size], 0, width * (1 - crop_pct))
        top = tf.compat.v1.random_uniform([batch_size], 0, height * (1 - crop_pct))
        crop_transform = tf.stack([
          crop_pct,
          tf.zeros([batch_size]), top,
          tf.zeros([batch_size]), crop_pct, left,
          tf.zeros([batch_size]),
          tf.zeros([batch_size])
        ], 1)

        coin = tf.less(tf.compat.v1.random_uniform([batch_size], 0, 1.0), crop_probability)
        transforms.append(tf.where(tf.expand_dims(coin, 1), crop_transform, tf.expand_dims(identity, 0)))

    if transforms:
        images = tfa.image.transform(
          images,
          tfa.image.transform_ops.compose_transforms(transforms),
          interpolation='BILINEAR') # or 'NEAREST'
        masks = tfa.image.transform(
          masks,
          tfa.image.transform_ops.compose_transforms(transforms),
          interpolation='NEAREST') # or 'NEAREST'

    def cshift(values): # Circular shift in batch dimension
        return tf.concat([values[-1:, ...], values[:-1, ...]], 0)

    if mixup > 0:
        beta = tf.distributions.Beta(mixup, mixup)
        lam = beta.sample(batch_size)
        ll = tf.expand_dims(tf.expand_dims(tf.expand_dims(lam, -1), -1), -1)
        images = ll * images + (1 - ll) * cshift(images)

    return images, masks


def augment_xy_3d(images,
		resize=None, # (width, height) tuple or None
		horizontal_flip=False,
		vertical_flip=False,
		rotate=0, # Maximum rotation angle in degrees
		crop_probability=0, # How often we do crops
		crop_min_percent=0.6, # Minimum linear dimension of a crop
		crop_max_percent=1.,  # Maximum linear dimension of a crop
		mixup=0):  # Mixup coeffecient, see https://arxiv.org/abs/1710.09412.pdf
    images = tf.squeeze(images, axis=[-1])
    images = augment(images,
		     resize=resize, 
		     horizontal_flip=horizontal_flip, 
                     vertical_flip=vertical_flip, 
		     rotate=rotate, 
		     crop_probability=crop_probability, 
		     crop_min_percent=crop_min_percent,
		     crop_max_percent=crop_max_percent,
		     mixup=mixup)
    images = tf.expand_dims(images, axis=-1)
    return images


def augment_xy_nd(images,
		resize=None, # (width, height) tuple or None
		horizontal_flip=False,
		vertical_flip=False,
		rotate=0, # Maximum rotation angle in degrees
		crop_probability=0, # How often we do crops
		crop_min_percent=0.6, # Minimum linear dimension of a crop
		crop_max_percent=1.,  # Maximum linear dimension of a crop
		mixup=0):  # Mixup coeffecient, see https://arxiv.org/abs/1710.09412.pdf
    img_shape = images.get_shape().as_list()
    images = tf.reshape(images, [img_shape[0], img_shape[1], img_shape[2], -1])
    images = augment(images,
		     resize=resize, 
		     horizontal_flip=horizontal_flip, 
                     vertical_flip=vertical_flip, 
		     rotate=rotate, 
		     crop_probability=crop_probability, 
		     crop_min_percent=crop_min_percent,
		     crop_max_percent=crop_max_percent,
		     mixup=mixup)
    images = tf.reshape(images, img_shape)
    return images


def outline_augment(outlines, resize=False,  # scaling
                    resize_std=0.1,  # scale factor follows N(1, resize_std^2)
                    horizontal_flip=False,
                    vertical_flip=False,
                    rotate=0,
                    noise_probability=0.0, # the probability of noise adding to each point
                    noise_std=0.1,
                    translate=None
                    ):

    # Here we assume the input outlines is centered. 
    # My experiments showed that casting on GPU improves training performance
    if outlines.dtype != tf.float32:
        outlines =tf.to_float(outlines)
  
    with tf.name_scope('augmentation'):
        shp = tf.shape(outlines)
        batch_size, height, width = shp[0], shp[1], shp[2]
        # width = tf.cast(width, tf.float32)
        # height = tf.cast(height, tf.float32)

        if resize is not None:
            resize_factors = tf.random_normal([batch_size, 1, 1], mean=1.0, stddev=resize_std)
            outlines=outlines * resize_factors 
        
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        if horizontal_flip:
            coin = tf.less(tf.random_uniform([batch_size, 1, 1], 0, 1.0), 0.5)
            horn_flip = tf.to_float(coin) * 2 - 1
        else:
            horn_flip = tf.ones([batch_size, 1, 1]) 

        if vertical_flip:
            coin = tf.less(tf.random_uniform([batch_size, 1, 1], 0, 1.0), 0.5)
            vert_flip = tf.to_float(coin) * 2 - 1
        else:
            vert_flip = tf.ones([batch_size, 1, 1])

        horn_vert_flip = tf.concat([horn_flip, vert_flip], 1)
        outlines = outlines * horn_vert_flip

        if rotate > 0:
            angle_rad = rotate / 180 * math.pi
            angles = tf.random_uniform([batch_size, 1], -angle_rad, angle_rad)
            rotation_matrix = tf.concat([tf.cos(angles), tf.sin(angles), tf.cos(angles), tf.sin(angles)], 1)
            rotation_matrix = tf.reshape(rotation_matrix, [-1, 2, 2])
            outlines = tf.matmul(rotation_matrix, outlines)
        
        if noise_probability > 0.0:
            noise = tf.random_normal([batch_size, height, width], mean=0.0, stddev=noise_std)
            outlines = outlines + noise
        
        if translate is not None:
            rand_translate = tf.random_uniform([batch_size, height, 1], -translate, translate)
            outlines = outlines + rand_translate
      
    return outlines
        

def main():
    # These can be any tensors of matching type and dimensions.
    images = tf.random.uniform(dtype=tf.dtypes.float32, shape=(2, 512, 512, 1))
    masks = tf.random.uniform(dtype=tf.dtypes.float32, shape=(2, 512, 512, 1))

    images = augment(images, masks, crop_shape=(256, 256), horizontal_flip=True, vertical_flip=True, rotate=15, crop_probability=0.8, mixup=0)
    
    # outlines = tf.placeholder(tf.float32, shape=(None, 2, 2000))
    # outlines = outline_augment(outlines, resize=True, scale_std=0.1, horizontal_flip=True, vertical_flip=True, retate=90, noise_probability=0.5, noise_std=0.1, translate=1)


if __name__ == '__main__':
    main()