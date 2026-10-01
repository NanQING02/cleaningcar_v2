import csv
import os
import tempfile
import unittest
from pathlib import Path

from web.server import EvidenceCsvCache


class EvidenceCsvCacheTests(unittest.TestCase):
    def test_append_is_loaded_incrementally_and_manifest_lookup_uses_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'event_index.csv'
            with path.open('w', encoding='utf-8-sig', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['event_id', 'manifest'])
                writer.writeheader()
                writer.writerow({'event_id': 'a', 'manifest': 'a.json'})

            cache = EvidenceCsvCache()
            self.assertEqual([row['event_id'] for row in cache.rows(path)], ['a'])

            with path.open('a', encoding='utf-8', newline='') as handle:
                csv.DictWriter(handle, fieldnames=['event_id', 'manifest']).writerow(
                    {'event_id': 'b', 'manifest': 'b.json'}
                )

            self.assertEqual([row['event_id'] for row in cache.rows(path)], ['a', 'b'])
            self.assertEqual(cache.manifest_for(path, 'b'), 'b.json')
            self.assertEqual(
                [row['event_id'] for row in cache.search(path, '', 1)],
                ['b'],
            )
            self.assertEqual(
                [row['event_id'] for row in cache.search(path, 'a.json', 10)],
                ['a'],
            )

    def test_truncated_file_rebuilds_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'event_index.csv'
            path.write_text('event_id,manifest\na,a.json\n', encoding='utf-8')
            cache = EvidenceCsvCache()
            self.assertEqual(len(cache.rows(path)), 1)

            original_stat = path.stat()

            path.write_text('event_id,manifest\nc,c.json\n', encoding='utf-8')
            os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            self.assertEqual([row['event_id'] for row in cache.rows(path)], ['c'])

    def test_larger_replacement_is_not_treated_as_append(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'event_index.csv'
            path.write_text('event_id,manifest\na,a.json\n', encoding='utf-8')
            cache = EvidenceCsvCache()
            self.assertEqual([row['event_id'] for row in cache.rows(path)], ['a'])

            path.write_text(
                'event_id,manifest\nreplacement,replacement.json\nsecond,second.json\n',
                encoding='utf-8',
            )

            self.assertEqual(
                [row['event_id'] for row in cache.rows(path)],
                ['replacement', 'second'],
            )
