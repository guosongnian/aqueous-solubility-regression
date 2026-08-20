"""不下载数据的快速单元测试。"""

import unittest

from sklearn.linear_model import Ridge

from solubility_regression import model_family


class SolubilitySmokeTest(unittest.TestCase):
    def test_model_name_mapping(self) -> None:
        self.assertEqual(model_family(Ridge()), "Ridge")


if __name__ == "__main__":
    unittest.main()
