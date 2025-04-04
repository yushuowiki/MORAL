import unittest
from datasets import fb15k237

class TestFB15k237(unittest.TestCase):
    def test_info(self):
        num_entities, num_relations = fb15k237.get_info()
        self.assertEqual(num_entities, 14505)
        self.assertEqual(num_relations, 237)

    def test_dicts(self):
        entity_to_id, relation_to_id = fb15k237.get_dicts()
        self.assertEqual(len(entity_to_id), 14505)
        self.assertEqual(len(relation_to_id), 237)
    
    def test_rawdata(self):
        train, test, valid = fb15k237.get_raw_data()
        self.assertEqual(len(train), 272115)
        self.assertEqual(len(test), 20438)
        self.assertEqual(len(valid), 17526)

    def test_datasets(self):
        train = fb15k237.get_datasets()
        test = fb15k237.get_datasets(mode='test')
        valid = fb15k237.get_datasets(mode='valid')
        self.assertEqual(len(train), 272115)
        self.assertEqual(len(test), 20438)
        self.assertEqual(len(valid), 17526)
    
    def test_torch_dataset(self):
        pass

    def setUp(self):
        pass
    
    def tearDown(self):
        pass

    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass