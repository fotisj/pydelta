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

    def _update_from(self, d):
        """
        Internal helper to update inner dictionary 'with semantics'. This will
        append rather then overwrite existing md fields if they are in a
        specified list. Clients should use :meth:`update` or the constructor
        instead.

        :param dict d: Dictionary to update from.
        """
        if isinstance(d, dict):
            appendables = ('normalization',)
            d2 = dict(d)

            for field in appendables:
                if field in d and field in self.__dict__:
                    d2[field] = self.__dict__[field] + d[field]

            self.__dict__.update(d2)
        else:
            self.__dict__.update(d)


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
                self._update_from(arg.__dict__)
            elif "metadata" in dir(arg) and isinstance(arg.metadata, Metadata):
                self._update_from(arg.metadata.__dict__)
            elif isinstance(arg, str):
                self._update_from(json.loads(arg))
            else:
                self._update_from(arg)
        self._update_from(kwargs)

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
        """
        Returns a JSON string containing this metadata object's contents.

        :param **kwargs: Arguments passed to :func:`json.dumps`
        """
        return json.dumps(self.__dict__, **kwargs)
