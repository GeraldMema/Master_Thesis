class Data_Preprocessing:

    def __init__(self, data_set):
        self.data = data_set
        self.process()

    def process(self):
        y = self.data.iloc[:, -1:]
        X = self.data.drop(self.data.iloc[:, -1:], axis=1)

        # data analysis
