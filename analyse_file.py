import h5py


def allkeys(obj):
    'Recursively find all keys in an h5py.Group.'
    keys = (obj.name,)
    if isinstance(obj, h5py.Group):
        for key, value in obj.items():
            if isinstance(value, h5py.Group):
                keys = keys + allkeys(value)
            else:
                keys = keys + (value.name,)
    return keys

file_path = 'data/cw_hdf5/signal0_0.hdf5'
#file_path = './example/00054c878.hdf5'

with h5py.File(file_path, 'r') as file:
    all_keys = allkeys(file)
    print('\n'.join(all_keys))

    print('------------')

    for key in all_keys:
        obj = file[key]
        if isinstance(obj, h5py.Dataset):
            print(f'{key} -> {obj.shape}')
    #print(file['/0517ef7fe/H1/SFTs'][0][0])
    #print(file['/0517ef7fe/H1/SFTs'][0][1])
    #for i in range(360):
    #    print(file['/0517ef7fe/frequency_Hz'][i])
