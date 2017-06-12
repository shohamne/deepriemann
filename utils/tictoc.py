# tic toc functions ala Matlab
# https://gist.github.com/tylerhartley/5174230

import time


def tic(tag=None):
    '''Start timer function.
    tag = used to link a tic to a later toc. Can be any dictionary-able key.
    '''
    global TIC_TIME
    if tag is None:
        tag = 'default'

    try:
        TIC_TIME[tag] = time.time()
    except NameError:
        TIC_TIME = {tag: time.time()}


def toc(tag=None, save=False, fmt=False):
    '''Timer ending function.
    tag - used to link a toc to a previous tic. Allows multipler timers, nesting timers.
    save - if True, returns float time to out (in seconds)
    fmt - if True, formats time in H:M:S, if False just seconds.
    '''
    global TOC_TIME
    template = 'Elapsed time is:'
    if tag is None:
        tag = 'default'
    else:
        template = '%s - ' % tag + template

    try:
        TOC_TIME[tag] = time.time()
    except NameError:
        TOC_TIME = {tag: time.time()}

    if TIC_TIME:
        d = (TOC_TIME[tag] - TIC_TIME[tag])

        if fmt:
            print template + ' %s' % time.strftime('%H:%M:%S', time.gmtime(d))
        else:
            print template + ' %f seconds' % (d)

        if save: return d

    else:
        print "no tic() start time available. Check global var settings"