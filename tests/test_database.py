import tempfile
import unittest
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from database import (
    Base,
    DatabaseConnection,
    Topic,
    create_database_engine,
    database_connection,
)


class SQLAlchemyDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.engine = create_database_engine(Path(self.temp_dir.name) / "test.db")
        Base.metadata.create_all(self.engine)

    def tearDown(self):
        self.engine.dispose()
        self.temp_dir.cleanup()

    def test_existing_sql_queries_work_with_sqlalchemy_and_model_mapping(self):
        with database_connection(self.engine) as connection:
            result = connection.execute(
                "INSERT INTO topics(title, description) VALUES (?, ?)",
                ("Cells", "Cell biology"),
            )
            topic_id = result.lastrowid
            row = connection.execute(
                "SELECT id, title FROM topics WHERE id = ?", (topic_id,)
            ).fetchone()

        self.assertEqual(row["title"], "Cells")
        self.assertEqual(row[0], topic_id)
        self.assertEqual(dict(row), {"id": topic_id, "title": "Cells"})
        with database_connection(self.engine) as connection:
            update_result = connection.execute(
                "UPDATE topics SET title = ? WHERE id = ?", ("Cells Updated", topic_id)
            )
            self.assertEqual(update_result.rowcount, 1)

    def test_declarative_models_share_the_app_schema(self):
        with Session(self.engine) as session:
            session.add(Topic(title="Genetics", description="Inheritance"))
            session.commit()
            topic = session.scalar(select(Topic).where(Topic.title == "Genetics"))

        self.assertIsNotNone(topic)
        self.assertEqual(topic.description, "Inheritance")

    def test_database_context_rolls_back_on_error(self):
        with self.assertRaises(RuntimeError):
            with database_connection(self.engine) as connection:
                connection.execute(
                    "INSERT INTO topics(title, description) VALUES (?, ?)",
                    ("Temporary", "Rollback"),
                )
                raise RuntimeError("force rollback")

        with database_connection(self.engine) as connection:
            count = connection.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
