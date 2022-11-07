try:
    from ._C import * # noqa
except: # noqa
    try:
        from up_extensions._C import * # noqa
    except: # noqa
        pass
