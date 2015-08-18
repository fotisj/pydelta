# -*- coding: utf-8 -*-
"""
Contains utility classes and functions.
"""

import json
from collections.abc import Mapping

class MetadataException(Exception):
    pass

class Metadata(Mapping):
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
            Metadata(lower_case=True, sorted=True, words=5000)
        """
        self.update(*args, **kwargs)

    def _update_from(self, d):
        """
        Internal helper to update inner dictionary 'with semantics'. This will
        append rather then overwrite existing md fields if they are in a
        specified list. Clients should use :meth:`update` or the constructor
        instead.

        Args:
            d (dict): Dictionary to update from.
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

    # maybe inherit from mappingproxy?
    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)


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

        Args:
            filename (str): The name of the metadata file, or of the file to which a sidecar metadata filename exists
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

        Args:
            filename (str): Name of the metadata file or the source file
            **kwargs: are passed on to :func:`json.dump`
        """
        metafilename = self.metafilename(filename)
        with open(metafilename, "wt", encoding="utf-8") as f:
            json.dump(self.__dict__, f, **kwargs)

    def __repr__(self):
        return type(self).__name__ + '(' + \
                ', '.join(str(key) + '=' + repr(self.__dict__[key])
                        for key in sorted(self.__dict__.keys())) + ')'

    def to_json(self, **kwargs):
        """
        Returns a JSON string containing this metadata object's contents.

        Args:
            **kwargs: Arguments passed to :func:`json.dumps`
        """
        return json.dumps(self.__dict__, **kwargs)


class DocumentDescriber:
    """
    DocumentDescribers are able to extract metadata from the document IDs of a corpus.

    The idea is that a :class:`Corpus` contains some sort of document name
    (e.g., original filenames), however, some components would be interested in
    information inferred from metadata. A DocumentDescriber will be able to
    produce this information from the document name, be it by inferring it
    directly (e.g., using some filename policy) or by using an external
    database.

    This base implementation expects filenames of the format
    "Author_Title.ext" and returns author names as groups and titles as
    in-group labels.

    The :class:`DefaultDocumentDescriber` adds author and title shortening, and we plan
    a metadata based :class:`TableDocumentDescriber` that uses an external metadata table.
    """

    def group_name(self, document_name):
        """
        Returns the unique name of the group the document belongs to.

        The default implementation returns the part of the document name before
        the first ``_``.
        """
        return document_name.split('_')[0]

    def item_name(self, document_name):
        """
        Returns the name of the item within the group.

        The default implementation returns the part of the document name after
        the first ``_``.
        """
        return document_name.split('_')[1]

    def group_label(self, document_name):
        """
        Returns a (maybe shortened) label for the group, for display purposes.

        The default implementation just returns the :meth:`group_name`.
        """
        return self.group_name(document_name)

    def item_label(self, document_name):
        """
        Returns a (maybe shortened) label for the item within the group, for
        display purposes.

        The default implementation just returns the :meth:`item_name`.
        """
        return self.item_name(document_name)

    def label(self, document_name):
        """
        Returns a label for the document (including its group).
        """
        return self.group_label(document_name) + ': ' + self.item_label(document_name)

    def groups(self, documents):
        """
        Returns the names of all groups of the given list of documents.
        """
        return { self.group_name(document) for document in documents }

class DefaultDocumentDescriber(DocumentDescriber):

    def group_label(self, document_name):
        """
        Returns just the author's surname.
        """
        return self.group_name(document_name).split(',')[0]

    def item_label(self, document_name):
        """
        Shortens the title to a meaningful but short string.
        """
        junk = ["Ein", "Eine", "Der", "Die", "Das"]
        title = self.item_name(document_name)
        title_parts = title.split(" ")
        #getting rid of file ending .txt
        if ".txt" in title_parts[-1]:
            title_parts[-1] = title_parts[-1].split(".")[0]
        #getting rid of junk at the beginning of the title
        if title_parts[0] in junk:
            title_parts.remove(title_parts[0])
        t = " ".join(title_parts)
        if len(t) > 25:
            return t[0:24] + '…'
        else:
            return t
