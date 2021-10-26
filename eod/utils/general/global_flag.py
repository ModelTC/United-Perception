

class Aligned():
    def __init__(self):
        self.aligned = False
        self.offset = 1


class Fp16Flag():
    def __init__(self):
        self.fp16 = False


class IterBase():
    def __init__(self):
        self.flag = False


class Mosaic_p():
    def __init__(self):
        self.flag = True


ITER_BASE_FLAG = IterBase()
FP16_FLAG = Fp16Flag()
ALIGNED_FLAG = Aligned()
