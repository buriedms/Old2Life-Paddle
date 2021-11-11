import paddle
from paddle.vision.transforms import BaseTransform
class Lambda(BaseTransform):
    def __init__(self, lambd):
        if not callable(lambd):
            raise TypeError("Argument lambd should be callable, got {}".format(repr(type(lambd).__name__)))
        self.lambd = lambd

    def _apply_image(self, img):
        return self.lambd(img)