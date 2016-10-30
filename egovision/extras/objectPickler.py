
class ObjectPickler:
    
    def __init__(self):
        pass

    @classmethod
    def save(self, obj, filename):
        """
        
        Saving the object object as a python pickle file
        
        """
        import gzip
        import cPickle
        fout = gzip.GzipFile(filename, 'w')
        cPickle.dump(obj,fout)
        fout.close()
        return True
    
    @classmethod
    def load(self, cls, filename ):
        """
        
        Initialize a datamanager using a pickle file

        :ivar String filename: Pickle file name with extension.

        :returns: [HandDetector] HandDetector previously saved
        
        """
        import gzip
        import cPickle
        fout = gzip.GzipFile(filename, 'r')
        obj = cPickle.load(fout)
        fout.close()
        if obj != None:
            if isinstance(obj, cls):
                return obj
            else:
                from exceptions import ImportError
                raise ImportError("Pickle is not " + cls.__class__)
        else:
            from exceptions import ImportError
            raise ImportError("Impossible to load the pickle file")
        return obj
