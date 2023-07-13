from spektral.data import Dataset, Graph


class MyDataset(Dataset):

    def __init__(self, input_vector_one_hot_encoding, contact_matrix, expected_results, **kwargs):
        self.__length = len(input_vector_one_hot_encoding)
        self.__gcn_input_vector_one_hot_encoding = input_vector_one_hot_encoding
        self.__contact_matrix = contact_matrix
        self.__expected_results = expected_results

        super().__init__(**kwargs)

    def read(self):
        return [Graph(x=self.__gcn_input_vector_one_hot_encoding, a=self.__contact_matrix, y=self.__expected_results)]

    def size(self):
        return self.__length
