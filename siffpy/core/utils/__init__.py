from siffpy.core.utils.im_params.im_params import ImParams  # noqa: F401

def warn_for_mroi(self):
    """
    Warn the user that MROI is not yet implemented explicitly
    """
    if self.im_params.RoiManager.mroiEnable:
        import warnings
        warnings.warn(
            'These data use the mROI functionality, which does not have explicit \
            support in SiffPy yet. Please be aware that the shapes of your data may \
            not be as expected.'
        )