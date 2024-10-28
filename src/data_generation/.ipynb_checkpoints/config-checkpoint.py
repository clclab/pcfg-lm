class Config:
    def __init__(self, **kwargs):
        for kwarg, val in kwargs.items():
            setattr(self, kwarg, val)

    def __repr__(self):
        representation = ""
        
        max_len = max(map(len, self.__dict__.keys()))
        for key, value in self.__dict__.items():
            str_value = str(value).split("\n")[0][:20]
            if len(str(value).split("\n")) > 1 or len(str(value).split("\n")[0]) > 20:
                str_value += " [..]"
            representation += (f"{key:<{max_len+3}}{str_value}\n")
            
        return representation