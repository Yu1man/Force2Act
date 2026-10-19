import h5py

def get_demo_names(hdf5_path: str):
    with h5py.File(hdf5_path, "r") as f:

        if "data" not in f:
            raise KeyError("HDF5 file does not contain a 'data' group")

        data = f["data"]

        demos = [name for name in data.keys() if name.startswith("demo_")]
        demos = sorted(demos, key=lambda name: int(name.split("_")[-1]))    # ["demo", "23"] -> [-1] = "23" -> int(23)

        return demos


def _decode_names(values):
    names = []

    for value in values:
        value = value.decode("utf-8")
        names.append(value)

    return names

def get_train_val_names(hdf5_path: str):
    with h5py.File(hdf5_path, "r") as f:

        if "mask" not in f:
            raise KeyError("HDF5 file does not contain a 'mask' group")

        data = f["mask"]

        if "train" not in data:
            raise KeyError("HDF5 file does not contain a 'train' mask")

        train = data["train"][()]
        train_names = _decode_names(train)
        train_names = sorted(train_names, key=lambda name: int(name.split('_')[-1]))

        if not train_names:
            raise ValueError("Training set is empty")

        if "valid" not in data:
            raise KeyError("HDF5 file does not contain a 'valid' mask")


        valid = data["valid"][()]
        valid_names = _decode_names(valid)
        valid_names = sorted(valid_names, key=lambda name: int(name.split('_')[-1]))

        if not valid_names:
            raise ValueError("Validation set is empty")

        overlap = set(train_names) & set(valid_names)

        if overlap:
            raise ValueError(f"{overlap} existed in both train and validation set")

        if "data" not in f:
            raise KeyError("HDF5 file does not contain a 'data' group")

        demo_names = [name for name in f["data"].keys() if name.startswith("demo_")]

        missing_train = set(train_names) - set(demo_names)
        missing_valid = set(valid_names) - set(demo_names)

        if missing_train:
            raise ValueError(f"{missing_train} from the training set does not exist in the demo")

        if missing_valid:
            raise ValueError(f"{missing_valid} from the validation set does not exist in the demo")

    return train_names, valid_names





if __name__ == "__main__":
    hdf5_path="/home/yui/Desktop/forcevam_karl/datasets/robosuite/door_push_pull.hdf5"
    with h5py.File(hdf5_path, "r") as f:
        mask = f["mask"]

        print(mask.keys())
        print(type(mask["train"]))
        print(mask["train"][()])
        print(mask["valid"][()])

    with h5py.File(hdf5_path, "r") as f:
        train = _decode_names(
            f["mask"]["train"][()]
        )

        print(train[:10])
        print(type(train[0]))

        train_names, valid_names = get_train_val_names(hdf5_path)

        print(len(train_names))
        print(train_names[:10])

        print(len(valid_names))
        print(valid_names[:10])