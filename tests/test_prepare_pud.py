import unittest
from tools.prepare_pud import parse_sentences

class PUDPreparationTests(unittest.TestCase):
    def test_preserves_id_text_and_excludes_annotations(self):
        blob='# sent_id = n1\n# text = نص عربي\n1\tنص\tlemma\n\n# sent_id = w2\n# text = Other.\n'.encode()
        self.assertEqual(parse_sentences(blob),[{'id':'n1','text':'نص عربي'},{'id':'w2','text':'Other.'}])
    def test_incomplete_sentence_metadata_rejected(self):
        with self.assertRaises(ValueError):parse_sentences(b'# sent_id = n1\n1\tx\n')
