import unittest
import numpy as np
import ds_toolbox.input_output.json_helpers as jh

test_dict = \
    {"Key_Uno": {"Ein": {"data": [1, 2, 3],
                         "labels": ["one", "two", "three"], },
                 "Zwei": {"data": [8.4, 2.2, np.nan],
                          "labels": ["l_1", "l_2", "x"],
                          "tags": "has_nans"},
                 "labels": ["label_Ein", "label_Zwei"], },
     "Key_Dos": "nada"}


class TestJsonHelpers(unittest.TestCase):

    def test_extract_values(self):

        actual_outputs = []
        test_keys = ["Ein", "labels", "data", "Key_Dos"]

        for test_key in test_keys:
            actual_outputs.append(jh.extract_values(test_dict, test_key))

        expected_outputs = [[{'data': [1, 2, 3], 'labels': ['one', 'two', 'three']}],
                            [['one', 'two', 'three'], ['l_1', 'l_2', 'x'], ['label_Ein', 'label_Zwei']],
                            [[1, 2, 3], [8.4, 2.2, np.nan]],
                            ["nada"]]

        self.assertEqual(expected_outputs, actual_outputs)

    def test_get_unique_keys_at_depth(self):

        actual_outputs = []
        for i in range(1, 4):
            actual_output = jh.get_unique_keys_at_depth(test_dict, i)
            actual_outputs.append(actual_output)

        expected_outputs = [['Key_Uno', 'Key_Dos'],
                            ['Ein', 'Zwei', 'labels'],
                            ['data', 'labels', 'tags']]

        self.assertEqual(expected_outputs, actual_outputs)

    def test_get_dict_depth(self):
        self.assertEqual(jh.get_dict_depth(test_dict), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
