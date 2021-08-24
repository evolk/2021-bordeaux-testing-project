import logistic
import pytest
from math import isclose

@pytest.mark.parametrize("x, r, f", 
			[(0.1, 2.2, 0.198),
 			(0.2, 3.4, 0.544),
  			(0.75, 1.7,0.31875)])
def test_logistic_map(x,r,f):
    result = logistic.logistic_map(x,r)
    assert isclose(f, result)
