class MolGenException(Exception):
    pass


class ConfigException(MolGenException):
    pass


class FilterException(MolGenException):
    pass


class UndesirableMolecule(FilterException):
    pass
