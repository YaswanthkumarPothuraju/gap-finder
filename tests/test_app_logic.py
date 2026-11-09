import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import main
from starlette.requests import Request


class ProviderAccessTests(unittest.TestCase):
    def test_model_access_check_confirms_configured_model(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps({
            "data": [{"id": "openai/gpt-oss-120b"}]
        }).encode()
        with (
            patch.object(main, "API_KEY", "test-key"),
            patch.object(main, "API_PROVIDER", "groq"),
            patch.object(main, "API_BASE_URL", "https://api.groq.com/openai/v1"),
            patch.object(main, "CHAT_MODEL", "openai/gpt-oss-120b"),
            patch("main.urllib.request.urlopen", return_value=response),
        ):
            result = main.test_provider_model_access()

        self.assertIn("openai/gpt-oss-120b", result)

    def test_model_access_check_reports_model_not_available(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps({"data": [{"id": "another-model"}]}).encode()
        with (
            patch.object(main, "API_KEY", "test-key"),
            patch.object(main, "API_PROVIDER", "groq"),
            patch.object(main, "API_BASE_URL", "https://api.groq.com/openai/v1"),
            patch.object(main, "CHAT_MODEL", "openai/gpt-oss-120b"),
            patch("main.urllib.request.urlopen", return_value=response),
        ):
            with self.assertRaisesRegex(main.ModelError, "not listed as available"):
                main.test_provider_model_access()

    def test_quiz_provider_error_identifies_failing_stage_and_model(self):
        with (
            patch.object(main, "API_KEY", "test-key"),
            patch.object(main, "API_PROVIDER", "groq"),
            patch.object(main, "CHAT_MODEL", "openai/gpt-oss-120b"),
            patch.object(main, "chat_json", side_effect=main.ProviderAccessError("HTTP 403")),
        ):
            with self.assertRaisesRegex(
                main.ProviderAccessError,
                "Groq request for model openai/gpt-oss-120b failed during question generation",
            ):
                main.quiz_chat_json("prompt", "evidence", "question generation")


    def test_quiz_is_saved_with_local_questions_after_provider_access_denial(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = main.create_database_engine(Path(temp_dir) / "quiz.db")
            main.create_schema(engine)
            try:
                with main.database_connection(engine) as connection:
                    document_id = connection.execute(
                        "INSERT INTO documents(kind, filename, extracted_text) VALUES (?, ?, ?)",
                        ("notes", "notes.txt", "The cell membrane controls movement of substances."),
                    ).lastrowid
                    topic_id = connection.execute(
                        "INSERT INTO topics(title, description) VALUES (?, ?)",
                        ("Cell membrane", "Cell membrane function"),
                    ).lastrowid
                    chunk_id = connection.execute(
                        "INSERT INTO chunks(document_id, section, text) VALUES (?, ?, ?)",
                        (
                            document_id,
                            "Cell membrane",
                            "The cell membrane controls movement of substances into and out of the cell.",
                        ),
                    ).lastrowid
                    connection.execute(
                        "INSERT INTO chunk_topics(chunk_id, topic_id, similarity) VALUES (?, ?, ?)",
                        (chunk_id, topic_id, 0.8),
                    )

                scope = {
                    "type": "http",
                    "method": "POST",
                    "path": "/quiz/create",
                    "headers": [],
                    "query_string": b"",
                    "scheme": "http",
                    "server": ("testserver", 80),
                    "client": ("testclient", 1234),
                    "session": {"user": main.DEMO_USERNAME},
                }
                request = Request(scope)
                chunk = {
                    "id": chunk_id,
                    "section": "Cell membrane",
                    "text": (
                        "The cell membrane surrounds the cell and controls movement of "
                        "substances into and out of the cell."
                    ),
                    "similarity": 0.8,
                }
                with (
                    patch.object(main, "engine", engine),
                    patch.object(main, "retrieve_chunks", return_value=[chunk]),
                    patch.object(
                        main,
                        "generated_question",
                        side_effect=main.ProviderAccessError("Groq denied the request (HTTP 403)."),
                    ) as generate,
                ):
                    response = main.create_quiz(request, mode="targeted", topic_id=str(topic_id))

                self.assertEqual(response.status_code, 303)
                self.assertTrue(response.headers["location"].startswith("/quiz/"))
                self.assertEqual(generate.call_count, 1)
                self.assertIn("local fallback questions", request.session["notice"]["message"])
                with main.database_connection(engine) as connection:
                    saved_questions = connection.execute(
                        "SELECT stem FROM quiz_questions ORDER BY id"
                    ).fetchall()
                self.assertEqual(len(saved_questions), 3)
                self.assertEqual(len({row["stem"] for row in saved_questions}), 3)
            finally:
                engine.dispose()

    def test_quiz_passage_filter_rejects_syllabus_titles_weak_matches_and_cover_pages(self):
        relevant_passage = {
            "id": 8,
            "section": "Network Topologies",
            "text": (
                "In a bus topology all devices share one cable. In a star topology "
                "every device connects to a central switch."
            ),
            "similarity": 0.39,
        }
        short_cover = {
            "id": 9,
            "section": "General",
            "text": "Page 1 Computer Networks Study Notes",
            "similarity": 0.8,
        }
        weak_match = {**relevant_passage, "id": 10, "similarity": 0.04}

        self.assertEqual(
            main.usable_quiz_passages(
                {"title": "Computer Networks - Syllabus"}, [short_cover, relevant_passage]
            ),
            [],
        )
        self.assertEqual(
            main.usable_quiz_passages(
                {"title": "Network topologies"},
                [short_cover, weak_match, relevant_passage],
            ),
            [relevant_passage],
        )

    def test_fallback_answer_uses_a_sentence_relevant_to_the_topic(self):
        topic = {"title": "Network topologies"}
        chunk = {
            "id": 22,
            "section": "TCP/IP Model",
            "text": (
                "The TCP/IP model has four layers used on the internet. "
                "Network Topologies In a bus topology all devices share one cable."
            ),
        }

        question = main.fallback_question(topic, [chunk], index=0)

        self.assertIn("bus topology", question["answer_key"])
        self.assertNotIn("TCP/IP model has four layers", question["answer_key"])


class PassageRetrievalTests(unittest.TestCase):
    def test_syllabus_document_title_is_not_presented_as_a_quiz_topic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = main.create_database_engine(Path(temp_dir) / "topics.db")
            main.create_schema(engine)
            try:
                with main.database_connection(engine) as connection:
                    connection.execute(
                        "INSERT INTO topics(title, description) VALUES (?, ?)",
                        ("Computer Networks - Syllabus", "Computer Networks - Syllabus"),
                    )
                    connection.execute(
                        "INSERT INTO topics(title, description) VALUES (?, ?)",
                        ("Network topologies", "Network topologies"),
                    )
                    topics = main.leaf_topics(connection)

                self.assertEqual([topic["title"] for topic in topics], ["Network topologies"])
            finally:
                engine.dispose()

    def test_topic_retrieval_falls_back_to_unmapped_notes_and_indexes_section(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = main.create_database_engine(Path(temp_dir) / "retrieval.db")
            main.create_schema(engine)
            try:
                with main.database_connection(engine) as connection:
                    notes_id = connection.execute(
                        "INSERT INTO documents(kind, filename, extracted_text) VALUES (?, ?, ?)",
                        ("notes", "networks.txt", "Network notes"),
                    ).lastrowid
                    syllabus_id = connection.execute(
                        "INSERT INTO documents(kind, filename, extracted_text) VALUES (?, ?, ?)",
                        ("syllabus", "syllabus.txt", "UDP"),
                    ).lastrowid
                    past_paper_id = connection.execute(
                        "INSERT INTO documents(kind, filename, extracted_text) VALUES (?, ?, ?)",
                        ("past_paper", "exam.txt", "UDP UDP UDP transport protocol"),
                    ).lastrowid
                    udp_topic_id = connection.execute(
                        "INSERT INTO topics(title, description) VALUES (?, ?)",
                        ("UDP", "UDP transport protocol"),
                    ).lastrowid
                    unrelated_topic_id = connection.execute(
                        "INSERT INTO topics(title, description) VALUES (?, ?)",
                        ("Routing", "Routing algorithms"),
                    ).lastrowid
                    udp_chunk_id = connection.execute(
                        "INSERT INTO chunks(document_id, section, text) VALUES (?, ?, ?)",
                        (
                            notes_id,
                            "UDP is a connectionless transport protocol",
                            "It has no handshake or retransmission, which keeps its latency low.",
                        ),
                    ).lastrowid
                    connection.execute(
                        "INSERT INTO chunk_topics(chunk_id, topic_id, similarity) VALUES (?, ?, ?)",
                        (udp_chunk_id, unrelated_topic_id, 0.3),
                    )
                    connection.execute(
                        "INSERT INTO chunk_topics(chunk_id, topic_id, similarity) VALUES (?, ?, ?)",
                        (udp_chunk_id, udp_topic_id, 0.01),
                    )
                    connection.execute(
                        "INSERT INTO chunks(document_id, section, text) VALUES (?, ?, ?)",
                        (syllabus_id, "UDP", "Syllabus-only text must not be used as evidence."),
                    )
                    connection.execute(
                        "INSERT INTO chunks(document_id, section, text) VALUES (?, ?, ?)",
                        (
                            past_paper_id,
                            "General",
                            "Compare UDP and TCP transport protocols.",
                        ),
                    )

                with (
                    patch.object(main, "engine", engine),
                    patch.object(main, "API_KEY", ""),
                ):
                    chunks = main.retrieve_chunks("UDP transport protocol", udp_topic_id)

                self.assertEqual(chunks[0]["id"], udp_chunk_id)
                self.assertTrue(all(chunk["kind"] != "syllabus" for chunk in chunks))
                self.assertGreater(chunks[0]["similarity"], 0.015)
                question = main.fallback_question(
                    {"title": "UDP"}, chunks, index=0
                )
                self.assertIn("UDP is a connectionless transport protocol", question["answer_key"])
            finally:
                engine.dispose()

    def test_ai_question_generation_receives_the_source_section_as_context(self):
        question_response = {
            "question_type": "short_answer",
            "stem": "What does UDP do?",
            "options": [],
            "answer_key": "UDP is connectionless.",
            "rubric": ["connectionless"],
            "source_chunk_ids": [9],
            "difficulty": "medium",
        }
        verifier_response = {"supported": True}
        chunk = {
            "id": 9,
            "section": "UDP is a connectionless transport protocol",
            "text": "It has no handshake or retransmission, which keeps its latency low.",
        }
        with (
            patch.object(main, "API_KEY", "test-key"),
            patch("main.chat_json", side_effect=[question_response, verifier_response]) as chat,
        ):
            main.generated_question({"title": "UDP"}, [chunk])

        evidence = json.loads(chat.call_args_list[0].args[1])["evidence"][0]["text"]
        self.assertIn("UDP is a connectionless transport protocol", evidence)
        self.assertIn("keeps its latency low", evidence)


class FallbackQuestionTests(unittest.TestCase):
    def test_fallback_question_uses_index_for_distinct_supported_prompts(self):
        topic = {"title": "Cell structure"}
        chunks = [
            {
                "id": 17,
                "text": (
                    "The cell membrane surrounds the cell and controls movement of "
                    "substances into and out of the cell."
                ),
            },
            {
                "id": 18,
                "text": "The nucleus contains genetic material and controls cell activities.",
            },
        ]

        questions = [
            main.fallback_question(topic, chunks, index)
            for index in range(3)
        ]

        self.assertEqual(len({question["stem"] for question in questions}), 3)
        self.assertEqual([question["source_chunk_ids"] for question in questions], [[17], [17], [17]])
        self.assertTrue(all(question["question_type"] == "short_answer" for question in questions))

    def test_local_quiz_generation_calls_fallback_with_question_index(self):
        with patch.object(main, "API_KEY", ""):
            question = main.generated_question(
                {"title": "Cell structure"},
                [{"id": 4, "text": "The nucleus contains the cell's genetic material."}],
                index=2,
            )

        self.assertIn("Explain one important point", question["stem"])
        self.assertEqual(question["source_chunk_ids"], [4])

    def test_provider_requests_include_user_agent(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps({"data": [{"id": "openai/gpt-oss-120b"}]}).encode()
        with (
            patch.object(main, "API_KEY", "test-key"),
            patch.object(main, "API_PROVIDER", "groq"),
            patch.object(main, "API_BASE_URL", "https://api.groq.com/openai/v1"),
            patch.object(main, "CHAT_MODEL", "openai/gpt-oss-120b"),
            patch("main.urllib.request.urlopen", return_value=response) as mock_urlopen,
        ):
            main.test_provider_model_access()
            request_obj = mock_urlopen.call_args[0][0]
            self.assertEqual(request_obj.get_header("User-agent"), main.USER_AGENT)

        with (
            patch.object(main, "API_KEY", "test-key"),
            patch.object(main, "API_PROVIDER", "groq"),
            patch.object(main, "API_BASE_URL", "https://api.groq.com/openai/v1"),
            patch("main.urllib.request.urlopen", return_value=response) as mock_urlopen,
        ):
            main.call_provider("chat/completions", {"model": "test"})
            request_obj = mock_urlopen.call_args[0][0]
            self.assertEqual(request_obj.get_header("User-agent"), main.USER_AGENT)

    def test_complete_plan_item_updates_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = main.create_database_engine(Path(temp_dir) / "plan.db")
            main.create_schema(engine)
            try:
                with main.database_connection(engine) as connection:
                    item_id = connection.execute(
                        """INSERT INTO plan_items(generation, day_number, minutes, priority, status, source_chunk_ids)
                        VALUES (1, 1, 30, 1, 'pending', '[]')"""
                    ).lastrowid

                scope = {
                    "type": "http",
                    "method": "POST",
                    "path": f"/plan/{item_id}/complete",
                    "headers": [],
                    "query_string": b"",
                    "scheme": "http",
                    "server": ("testserver", 80),
                    "client": ("testclient", 1234),
                    "session": {"user": main.DEMO_USERNAME},
                }
                request = Request(scope)
                with patch.object(main, "engine", engine):
                    response = main.complete_plan_item(request, item_id)
                self.assertEqual(response.status_code, 303)
                with main.database_connection(engine) as connection:
                    status = connection.execute(
                        "SELECT status FROM plan_items WHERE id = ?", (item_id,)
                    ).fetchone()["status"]
                self.assertEqual(status, "done")
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
