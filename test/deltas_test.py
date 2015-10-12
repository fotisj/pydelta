import delta as d
import os
from nose.tools import eq_
from math import log10, pow

testdir = None
c1000 = None


def setup_module():
    global testdir
    global c1000
    testdir = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)),
        'corpus3')
    c1000 = d.Corpus(testdir).get_mfw_table(1000)

def feq_(result, expected, msg=None, threshold=None):
    if threshold is None:
        threshold = pow(10, log10(expected)-2)
    if msg is None:
        msg = "{} != {}".format(result, expected)
    assert abs(expected - result) < threshold, msg

class Delta_Test:

    def check_function(self, function, expected_distance, expected_score=None):
        distances = function(c1000)
        sample = distances.at['Fontane,-Theodor_Der-Stechlin',
                              'Fontane,-Theodor_Effi-Briest']
        feq_(sample, expected_distance,
            "{} Stechlin/Effi distance is {} instead of {}!".format(
                function.name, sample, expected_distance))

        if expected_score is not None:
            feq_(expected_score, distances.simple_score(),
                "{} simple score is {} instead of expected {}!".format(
                    function.name, distances.simple_score(), expected_score))

    def burrows_test(self):
        self.check_function(d.registry.burrows, 0.7538867972199293)

    def linear_test(self):
        self.check_function(d.registry.linear, 1149.434663563308)

    def quadratic_test(self):
        self.check_function(d.registry.quadratic, 1102.845003724634)

    def eder_test(self):
        self.check_function(d.registry.eder, 0.3703309813454142)

    def cosine_delta_test(self):
        self.check_function(d.registry.cosine_delta, 0.6156353166442046)
