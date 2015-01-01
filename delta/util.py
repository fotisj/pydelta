"""
Contains utility classes and functions.
"""

import json

class MetadataException(Exception):
    pass

class Metadata(object):
    """
    A metadata record contains information about how a particular object of the
    pyDelta universe has been constructed, or how it will be manipulated.

    Metadata fields are simply attributes, and they can be used as such. 
    """

    def __init__(self, *args, **kwargs):
        """
        Create a new metadata instance. Arguments will be passed on to :meth:`update`.

        Examples:
        >>> m = Metadata(lower_case=True, sorted=False)
        >>> Metadata(m, sorted=True, words=5000)
        Metadata(words=5000, lower_case=True, sorted=True)
        """
        self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        """
        Updates this metadata record from the arguments. Arguments may be:

        * other :class:`Metadata` instances
        * objects that have ``metadata`` attribute
        * JSON strings
        * stuff that :class:`dict` can update from
        * key-value pairs of new or updated metadata fields
        """
        for arg in args:
            if isinstance(arg, Metadata):
                self.__dict__.update(arg.__dict__)
            elif "metadata" in arg and isinstance(arg.metadata, Metadata):
                self.__dict__.update(arg.metadata.__dict__)
            elif isinstance(arg, str):
                self.__dict__.update(json.loads(arg))
            else:
                self.__dict__.update(arg)
        self.__dict__.update(kwargs)

    @staticmethod
    def metafilename(filename):
        """
        Returns an appropriate metadata filename for the given filename.
        
        >>> Metadata.metafilename("foo.csv")
        'foo.csv.meta'
        >>> Metadata.metafilename("foo.csv.meta")
        'foo.csv.meta'
        """
        if filename.endswith('.meta'):
            return filename
        return filename + '.meta'

    @classmethod
    def load(cls, filename):
        """
        Loads a metadata instance from the filename identified by the argument.

        :param str filename: The name of the metadata file, or of the file to which a sidecar metadata filename exists
        """
        metafilename = cls.metafilename(filename)
        with open(metafilename, "rt", encoding="utf-8") as f:
            d = json.load(f)
            if isinstance(d, dict):
                return cls(**d)
            else:
                raise MetadataException("Could not load metadata from {file}: \n"
                        "The returned type is a {type}".format(file=metafilename, type=type(d)))

    def save(self, filename, **kwargs):
        """
        Saves the metadata instance to a JSON file.

        :param str filename: Name of the metadata file or the source file
        :param **kwargs: are passed on to :func:`json.dump`
        """
        metafilename = self.metafilename(filename)
        with open(metafilename, "wt", encoding="utf-8") as f:
            json.dump(self.__dict__, f, **kwargs)

    def __repr__(self):
        return type(self).__name__ + '(' + \
                ', '.join(str(key) + '=' + repr(value) 
                        for key, value in self.__dict__.items()) + ')'

    def to_json(self, **kwargs):
        return json.dumps(self.__dict__, **kwargs)
