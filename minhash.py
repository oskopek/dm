import numpy as np

MAX_HASH_FNS = 1024

R = 20
B = 2

assert R * B <= MAX_HASH_FNS

def hash_fns(values):
    return np.array([np.min(values), np.max(values)])

a = 2
b = 3
P = 15486869

# vector (signature of band) -> int
def hash_col(band_signature):
    # TODO: modulo later
    # TODO: use different hash fns later
    print band_signature.shape
    return sum((a * band_signature + b) % P)


def mapper(key, value):
    # key: None
    # value: one line of input file
    document, values_str = value.split(None, 1)
    document = int(document.split('_')[1])

    values = np.asarray(values_str.split(), dtype=np.uint64)
    hashes = hash_fns(values)

    signatures = np.zeros((B))
    for band_index in range(0, B):
        sig_vec = hashes[band_index * R : (band_index + 1) * R]
        signatures[band_index] = hash_col(sig_vec)
    str_signatures = [str(i) for i in signatures]
    sig_key = "-".join(str_signatures)
    yield sig_key, document

def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    num_of_docs = len(values)

    sorted(values) # TODO: Needed?
    for i in range(len(values)):
       for j in range(i, len(values)):
            yield values[i], values[j]
