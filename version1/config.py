from easydict import EasyDict as edict

# initialization
__C = edict()
cfg = __C

# seed value
__C.SEED = 2021
__C.DATASET = 'fakeConfig'

__C.EXP_PATH = './experiments'
__C.EXP_NAME = 'myProject'

__C.WRITE_FREQ = 10
