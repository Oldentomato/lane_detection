import pickle

def save_to_pickle(data_dict, dir):
    try:
        with open(dir, 'wb') as pickle_file:
            pickle.dump(data_dict, pickle_file)
        print(f'Data saved to {dir} successfully.')
    except Exception as e:
        print(f'Error while saving to {dir}: {str(e)}')


def load_from_pickle(dir):
    try:
        with open(dir, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
        print(f'Data loaded from {dir} successfully.')
        return data
    except Exception as e:
        print(f'Error while loading from {dir}: {str(e)}')
        return None