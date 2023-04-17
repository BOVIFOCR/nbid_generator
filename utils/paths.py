# -*- coding: latin-1 -*-

import enum
import random
import pathlib
import typing

Path = pathlib.PosixPath
Pathlike = typing.Union[str, Path]


class SampleLabelEnumMeta(enum.EnumMeta):
    def __normalize_attr_name(self, name: str) -> str:
        return name.upper()

    def __contains__(self, obj) -> bool:
        if isinstance(obj, str):
            value = self.__normalize_attr_name(obj)
            for member in self:
                if member.name == value:
                    return True
        return super().__contains__(obj)

    def __getattr__(self, name: str):
        return super().__getattr__(self.__normalize_attr_name(name))

    def __getitem__(self, name: str):
        return super().__getitem__(self.__normalize_attr_name(name))


class SampleLabel(enum.Enum, metaclass=SampleLabelEnumMeta):
    FRONT = enum.auto()
    BACK = enum.auto()

    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return self.__str__()

    def to_ptbr(self):
        return {
            'front': 'frente',
            'back': 'verso'
        }[str(self)]


class SamplePath(pathlib.PosixPath):
    def __new__(cls, path_args, synth_dir, init=True):
        if isinstance(path_args, Path):
            return cls._from_parts(path_args._parts)
        elif isinstance(path_args, str):
            return cls._from_parts([path_args])

        return cls._from_parts(path_args)

    def __add__(self, other):
        return other + str(self) if (
            isinstance(other, str)
        ) else super().__add__(other)

    def __reduce__(self):
        # For pickling
        return (self.__class__, tuple([list(self._parts)]+[self.synth_dir]))

    def __init__(self, path_args, synth_dir=None):
        self.synth_dir = synth_dir

    def sample_id(self, return_ext=False):
        return (self.stem, self.suffix) if return_ext else self.stem

    def labels_fpath(self):
        return self.synth_dir.path_json / (self.sample_id() + ".json")

    def anon_fpath(self):
        return (self.synth_dir.path_anon / self.name).with_suffix('.jpg')

    def warped_fpath(self):
        return (self.synth_dir.path_warped / self.name).with_suffix('.jpg')

    def tmpmask_fpath(self):
        return (self.synth_dir.path_mask / self.name).with_suffix('.jpg')


def get_repo_root_path():
    p = Path(__file__)
    while not (p.is_dir() and (p / '.git').exists()):
        p = p.parent
    return p


class SynthesisDir():
    path_repo_root = get_repo_root_path()
    path_input_root = path_repo_root / "synthesis_input"
    path_output_root = path_repo_root / "synthesis_output"

    def __init__(self, doc_side: SampleLabel, tmp_path=Path("/tmp")):
        self.path_input_base = self.path_input_root / str(doc_side)
        self.path_output_base = self.path_output_root / str(doc_side)

        self.path_input = self.path_input_base / "input"
        self.path_json = self.path_input_base / "labels"

        self.path_output = self.path_output_base / "synth_product"
        self.path_anon = self.path_output_base / "anon"
        self.path_warped = self.path_output_base / "warped"

        self.path_static = self.path_repo_root / "files"

        self.path_mpout = tmp_path / "mplock_out"
        self.path_mask = tmp_path / "synth_mask"

        self.mkdirs()

    def mkdirs(self):
        for name in dir(self):
            v = getattr(self, name)
            if isinstance(v, Path):
                for r in (self.path_output_base, self.path_mpout):
                    if (v == r) or (r in v.parents):
                        v.mkdir(exist_ok=True, parents=True)

    def new_sample_path(self, sample_id: str) -> SamplePath:
        return SamplePath(sample_id, synth_dir=self)

    def list_input_images(self, for_anon=False, randomize=False):
        fnames = map(self.new_sample_path, Path(self.path_input).iterdir())
        if randomize:
            fnames = list(fnames)
            random.shuffle(fnames)
        for input_fpath in fnames:
            _, file_ext = input_fpath.sample_id(return_ext=True)
    
            labels_file_exists = input_fpath.labels_fpath().exists()
            bg_labels_file_exists = Path(
                str(input_fpath.labels_fpath()).replace(".json", ".bg.json")
            ).exists()

            valid_fname_extension = file_ext.lower() in (".jpg", ".jpeg", ".png")

            back_file_exists = input_fpath.anon_fpath().exists()
            already_did_anon = back_file_exists and bg_labels_file_exists

            criteria = valid_fname_extension and labels_file_exists
            if for_anon:
                criteria = criteria and not already_did_anon
            else:
                criteria = criteria and already_did_anon

            if criteria:
                yield input_fpath
