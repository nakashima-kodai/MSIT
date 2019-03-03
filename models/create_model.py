def create_model(opt):
    if opt.model == 'MSIT':
        from .MSIT import MSIT
        model = MSIT()
        model.initialize(opt)
        model.setup()
    elif opt.model == 'pix2pix':
        from .pix2pix import pix2pix
        model = pix2pix()
        model.initialize(opt)
        model.setup()
    else:
        raise NotImplementedError('model [{}] is not found'.format(opt.model))

    print('model [{}] was created'.format(opt.model))
    return model
