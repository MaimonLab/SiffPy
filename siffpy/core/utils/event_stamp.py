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
        if hasattr(self, '_timestamp_epoch'):
            return self._timestamp_epoch
        raise AttributeError("No experiment_to_epoch conversion defined."
            + " Please define one using define_experiment_to_epoch.")
        
    def define_experiment_to_epoch(self, experiment_to_epoch : Callable):
        self._timestamp_epoch = experiment_to_epoch(self.timestamp)

    @property
    def timestamp_experiment(self):
        """ Alias of the timestamp which is already in experiment time """
        return self.timestamp
    
    def __repr__(self):
        return f"EventStamp(frame_number={self.frame_number}, text={self.text}, timestamp={self.timestamp})"

    def __str__(self):
        return f"EventStamp(frame_number={self.frame_number}, text={self.text}, timestamp={self.timestamp})"
    