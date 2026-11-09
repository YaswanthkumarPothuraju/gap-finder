import unittest

from gapfinder_core import (
    allocate_diagnostic_questions,
    allocate_plan,
    chunk_text,
    compute_mastery,
    lexical_similarity,
    mastery_label,
    parse_syllabus,
    parse_syllabus_tree,
)


class GapFinderCoreTests(unittest.TestCase):
    def test_syllabus_extracts_numbered_and_markdown_headings(self):
        topics = parse_syllabus("# Cell Biology\n1.1 Cell structure\n2. Genetics")
        self.assertEqual(topics, ["Cell Biology", "Cell structure", "Genetics"])

    def test_syllabus_keeps_heading_depth_for_parent_topics(self):
        topics = parse_syllabus_tree("# Biology\n## Cells\n### Membranes")
        self.assertEqual([topic["depth"] for topic in topics], [1, 2, 3])

    def test_chunking_preserves_section_and_overlaps_long_text(self):
        chunks = chunk_text("# Photosynthesis\n" + "chlorophyll energy " * 450, max_words=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["section"] == "Photosynthesis" for chunk in chunks))
        self.assertEqual(len(chunks[0]["text"].split()), 100)

    def test_lexical_similarity_ranks_related_evidence(self):
        related = lexical_similarity("cell membrane transport", "The cell membrane controls transport.")
        unrelated = lexical_similarity("cell membrane transport", "The solar system contains planets.")
        self.assertGreater(related, unrelated)

    def test_mastery_weights_difficulty_and_attempt_recency(self):
        mastery, count = compute_mastery([
            {"attempt_no": 1, "difficulty": "easy", "score": 0.0},
            {"attempt_no": 2, "difficulty": "hard", "score": 1.0},
        ])
        self.assertEqual(count, 2)
        self.assertGreater(mastery, 0.5)
        self.assertEqual(mastery_label(mastery, count), "needs more data")

    def test_plan_never_exceeds_available_time_and_reports_omissions(self):
        topics = [
            {"id": 1, "title": "A", "priority": 2, "minutes": 40},
            {"id": 2, "title": "B", "priority": 1, "minutes": 40},
        ]
        scheduled, retests, unscheduled = allocate_plan(topics, total_minutes=30, days=2)
        self.assertEqual(sum(item["minutes"] for item in scheduled + retests), 30)
        self.assertTrue(unscheduled)
        self.assertLessEqual(max(item["day"] for item in scheduled + retests), 2)

    def test_diagnostic_allocation_covers_topics_and_weights_extra_questions(self):
        topics = [
            {"id": 1, "exam_weight": 0.8},
            {"id": 2, "exam_weight": 0.2},
        ]
        allocated = allocate_diagnostic_questions(topics)
        counts = {topic["id"]: 0 for topic in topics}
        for topic, _ in allocated:
            counts[topic["id"]] += 1
        self.assertEqual(len(allocated), 10)
        self.assertEqual(set(counts), {topic["id"] for topic in topics})
        self.assertGreater(counts[1], counts[2])


if __name__ == "__main__":
    unittest.main()
