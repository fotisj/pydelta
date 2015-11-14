import delta as d
import os
from nose.tools import eq_

testdir = None


def setup_module():
    global testdir
    testdir = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)),
        'corpus3')


class FeatureGenerator_Test:

    def setup(self):
        self.gen = d.FeatureGenerator()

    def test_tokenize(self):
        assert list(self.gen.tokenize(["This is a", "simple test"])) \
            == ["This", "is", "a", "simple", "test"]

    def test_tokenize_letters(self):
        fg1 = d.FeatureGenerator(token_pattern=d.LETTERS_PATTERN)
        assert list(fg1.tokenize(["I don't like mondays."])) \
            == ["I", "don", "t", "like", "mondays"]

    def test_tokenize_words(self):
        fg1 = d.FeatureGenerator(token_pattern=d.WORD_PATTERN)
        assert list(fg1.tokenize(["I don't like mondays."])) \
            == ["I", "don't", "like", "mondays"]

    def test_count_tokens(self):
        result = self.gen.count_tokens(
            ["this is a test", "testing this generator"])
        assert result["this"] == 2
        assert result["generator"] == 1
        assert result.sum() == 7

    def test_get_name(self):
        assert self.gen.get_name('foo/bar.baz.txt') == 'bar.baz'

    def test_call(self):
        df = self.gen(testdir)
        eq_(df.und.sum(), 25738.0)

class Corpus_Test:

    def parse_test(self):
        corpus = d.Corpus(testdir)
        eq_(corpus.und.sum(), 25738.0)

    def mfw_test(self):
        corpus = d.Corpus(testdir)
        rel_corpus = corpus.get_mfw_table(0)
        eq_(rel_corpus.sum(axis=1).sum(), 9)



class Cluster_Test:

    def init_test(self):
        # FIXME
        corpus = d.Corpus(testdir).get_mfw_table(1000)
        deltas = d.registry.cosine_delta(corpus)
        hclust = d.Clustering(deltas)
        fclust = hclust.fclustering()
        print(fclust.describe())
        print(fclust.evaluate())
        assert fclust.data is not None

class Table_Describer_Test:

    def md_test(self):
        corpus = d.Corpus(testdir, document_describer=d.util.TableDocumentDescriber(testdir + '.csv', 'Author', 'Title'))
        assert corpus.document_describer.group_name(corpus.index[-1]) == 'Raabe'
