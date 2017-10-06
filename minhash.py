import numpy as np

MAX_HASH_FNS = 1024

B = 1
R = 2

def mapper(key, value):
    # key: None
    # value: one line of input file
    document, values_str = value.split(None, 1)
    document = int(document.split('_')[1])
    values = np.asarray(values_str.split(), dtype=np.uint64)
    yield 1, (document, np.min(values), np.max(values))


a = 2
b = 3
P = 15486869

# vector (signature of band) -> int
def hash_col(band_signature, num_of_docs):
    return sum(((a * band_signature + b) % P) % num_of_docs) % num_of_docs


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    num_of_docs = len(values)
    sig_matrix = np.zeros((B * R, num_of_docs))
    for j, val in enumerate(values):
        sig_matrix[:, j] = val[1:]
        #for i in range(0, B * R):
        # sig_matrix[i, j] = val[i+1]

    # TODO: Band hashing in mapper
    band_hashes = np.zeros((B, num_of_docs))
    for band_index in range(0, B):
        for col_index in range(0, num_of_docs):
            sig = sig_matrix[band_index * R : (band_index + 1) * R, col_index]
            band_hashes[band_index, col_index] = hash_col(sig, num_of_docs) # returns a number

    for pair1_index in range(num_of_docs):
       for pair2_index in range(num_of_docs):
           for band_index in range(B):
               yield 1, 1
