import holoviews as hv
from siffpy.sifftrac.log_interpreter.fictraclog import FictracLog

class LogToPlot(FictracLog):
    """
    Extends the FictracLog to include useful variables for plotting.
    Separated to avoid unnecessary Holoviews imports

    ATTRIBUTES
    ----------
    
    dataset : hv.Dataset



    """
    def __init__(self, *args, **kwargs):
        """
        For construction, but doesn't work right if there's
        already an existing FictracLog object that I just want
        to extend.
        """
        if 'FictracLog' in kwargs:
            # Inheritance from an existing FictracLog
            if not isinstance(kwargs['FictracLog'], FictracLog):
                raise TypeError()
            
            flog = kwargs['FictracLog']
            for attr_name, attr_val in flog.__dict__.items():
                setattr(self, attr_name, attr_val)
            

        else:
            super(FictracLog, self).__init__(*args, **kwargs)
            
        if 'image_time' in self._dataframe:
            self.dataset = hv.Dataset(
                self._dataframe, # Receives a copy, to keep this interactive
                'image_time', # kdims
                [ # vdims
                    ('integrated_position_lab_0','X'),
                    ('integrated_position_lab_1','Y'),
                ]
            )
            self.tdim = 'image_time'
        
        else:
            self.dataset = hv.Dataset(
                self._dataframe, # Receives a copy, to keep this interactive
                'timestamp', # kdims
                [ # vdims
                    ('integrated_position_lab_0','X'),
                    ('integrated_position_lab_1','Y')
                ]
            )
            self.tdim = 'timestamp'