class MolGenException(Exception):
    pass

class FilterException(MolGenException):
    pass


class UndesirableMolecule(FilterException):
    pass
