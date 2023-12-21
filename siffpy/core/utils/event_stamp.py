from typing import Optional, Callable

class EventStamp():
    """
    EventStamp is a dataclass that contains the information about events
    in SiffPy data ("Appended text")
    """

    def __init__(self,
        frame_number : int,
        text : str,
        timestamp: Optional[float] = None,
    ):
        self.frame_number = frame_number
        self.text = text
        self.timestamp = timestamp

    @property
    def timestamp_epoch(self):
        if hasattr(self, 'experiment_to_epoch'):
            return self.experiment_to_epoch(self.timestamp)
        raise AttributeError("No experiment_to_epoch conversion defined."
            + " Please define one using define_experiment_to_epoch.")
        
    def define_experiment_to_epoch(self, experiment_to_epoch : Callable):
        self.experiment_to_epoch = experiment_to_epoch

    @property
    def timestamp_experiment(self):
        """ Alias of the timestamp which is already in experiment time """
        return self.timestamp
    