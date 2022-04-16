class MolGenException(Exception):
    pass


class ConfigException(MolGenException):
    pass


class PreprocessingException(MolGenException):
    pass


class ConvertException(PreprocessingException):
    pass


class FilterException(PreprocessingException):
    pass


class UndesirableMolecule(FilterException):
    pass
