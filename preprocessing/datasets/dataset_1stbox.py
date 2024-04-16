from preprocessing.datasets.dataset import DatasetBasis

class Dataset1stBox(DatasetBasis):
    def __init__(self, path:str, box_size:int=None):
        DatasetBasis.__init__(self, path, box_size)

    def get_run_id(self, index):
        return self.input_names[index]