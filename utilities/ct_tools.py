


class CTTransforms:
    def __init__(
        self, 
        img_shape=None, 
        mu_water=0.192,  #  0.192 [1/cm] = 192 [1/m]
        mu_air=0.0,
        min_hu=-1024,
        max_hu=3072
    ):
        super().__init__()
        self.img_shape = img_shape
        self.min_hu, self.max_hu = min_hu, max_hu
        self.mu_water, self.mu_air = mu_water, mu_air
        self.width, self.center = (max_hu - min_hu), (min_hu + (max_hu - min_hu) / 2)
    
    def hu2mu(self, hu, min_hu=None, max_hu=None, do_clip=True):
        if do_clip:
            hu = self.clip_value(hu, min_hu, max_hu)
        mu = hu / 1000 * (self.mu_water - self.mu_air) + self.mu_water
        return mu

    def mu2hu(self, mu):
        hu = (mu - self.mu_water) / (self.mu_water - self.mu_air) * 1000
        return hu

    @staticmethod
    def clip_value(x, min_val, max_val):
        return x.clip(min_val, max_val) if hasattr(x, 'clip') else min(max(x, min_val), max_val)
    
    def normalize_hu(self, hu, min_hu=None, max_hu=None, do_clip=False):
        min_hu = self.min_hu if min_hu is None else min_hu
        max_hu = self.max_hu if max_hu is None else max_hu
        if do_clip:
            hu = self.clip_value(hu, min_hu, max_hu)
        norm_hu = (hu - min_hu) / (max_hu - min_hu)
        return norm_hu
    
    def denormalize_hu(self, norm_hu, min_hu=None, max_hu=None, do_clip=False):
        min_hu = self.min_hu if min_hu is None else min_hu
        max_hu = self.max_hu if max_hu is None else max_hu
        hu = (max_hu - min_hu) * norm_hu + min_hu
        if do_clip:
            hu = self.clip_value(hu, min_hu, max_hu)
        return hu
    
    def window_transform(self, hu_image, width=None, center=None, norm_to_255=False):
        ''' hu_image -> 0-1 normalization'''
        if width is None or center is None:
            width, center = self.width, self.center
        window_min = float(center) - 0.5 * float(width)
        win_image = (hu_image - window_min) / float(width)
        win_image[win_image < 0] = 0
        win_image[win_image > 1] = 1
        if norm_to_255:
            win_image = (win_image * 255).astype('float')
        return win_image

    def back_window_transform(self, win_image, width=None, center=None, norm_to_255=False):
        ''' 0-1 normalization -> hu_image'''
        if width is None or center is None:
            width, center = self.width, self.center
        window_min = float(center) - 0.5 * float(width)
        if norm_to_255:
            win_image = win_image / 255.
        hu_image = win_image * float(width) + window_min
        return hu_image
    
    def window_transform_torch(self, hu_image, width=None, center=None, norm_to_255=False):
        ''' hu_image -> 0-1 normalization'''
        if width is None or center is None:
            width, center = self.width, self.center
        window_min = float(center) - 0.5 * float(width)
        win_image = (hu_image - window_min) / float(width)
        win_image = win_image.float()
        win_image = win_image.clip(min=0.0, max=1.0)
        if norm_to_255:
            win_image = (win_image * 255).float()
        return win_image

    def back_window_transform_torch(self, hu_image, width=None, center=None, norm_to_255=False):
        if width is None or center is None:
            width, center = self.width, self.center
        return self.back_window_transform(hu_image, width, center, norm_to_255)

