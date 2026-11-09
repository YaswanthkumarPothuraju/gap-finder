from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
import re
from typing import Any

from sqlalchemy import (
    Float,
    ForeignKey,
    Index,
    Integer,
    CheckConstraint,
    String,
    Text,
    UniqueConstraint,
    URL,
    create_engine,
    event,
    text,
)
from sqlalchemy.engine import Connection, CursorResult, Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    uploaded_at: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    extracted_text: Mapped[str] = mapped_column(Text, nullable=False)


class Topic(Base):
    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("topics.id"))
    title: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''"))
    exam_weight: Mapped[float] = mapped_column(Float, nullable=False, server_default=text("1.0"))
    active: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (Index("idx_chunks_document", "document_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    page: Mapped[int | None] = mapped_column(Integer)
    section: Mapped[str] = mapped_column(String, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[str | None] = mapped_column(Text)


class ChunkTopic(Base):
    __tablename__ = "chunk_topics"

    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    topic_id: Mapped[int] = mapped_column(
        ForeignKey("topics.id", ondelete="CASCADE"), primary_key=True
    )
    similarity: Mapped[float] = mapped_column(Float, nullable=False)


class Quiz(Base):
    __tablename__ = "quizzes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False)
    mode: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )


class QuizQuestion(Base):
    __tablename__ = "quiz_questions"
    __table_args__ = (Index("idx_questions_quiz", "quiz_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    quiz_id: Mapped[int] = mapped_column(
        ForeignKey("quizzes.id", ondelete="CASCADE"), nullable=False
    )
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id"), nullable=False)
    question_type: Mapped[str] = mapped_column(String, nullable=False)
    stem: Mapped[str] = mapped_column(Text, nullable=False)
    options: Mapped[str] = mapped_column(Text, nullable=False)
    answer_key: Mapped[str] = mapped_column(Text, nullable=False)
    rubric: Mapped[str] = mapped_column(Text, nullable=False)
    source_chunk_ids: Mapped[str] = mapped_column(Text, nullable=False)
    difficulty: Mapped[str] = mapped_column(String, nullable=False)


class Response(Base):
    __tablename__ = "responses"
    __table_args__ = (Index("idx_responses_question", "question_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question_id: Mapped[int] = mapped_column(
        ForeignKey("quiz_questions.id", ondelete="CASCADE"), nullable=False
    )
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    correct: Mapped[int] = mapped_column(Integer, nullable=False)
    time_taken: Mapped[float] = mapped_column(Float, nullable=False, server_default=text("0"))
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    missed_concept: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''"))


class TopicScore(Base):
    __tablename__ = "topic_scores"
    __table_args__ = (UniqueConstraint("topic_id", "quiz_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id"), nullable=False)
    quiz_id: Mapped[int] = mapped_column(ForeignKey("quizzes.id"), nullable=False)
    mastery: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[str] = mapped_column(String, nullable=False)
    label: Mapped[str] = mapped_column(String, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)


class PlanItem(Base):
    __tablename__ = "plan_items"
    __table_args__ = (Index("idx_plan_generation", "generation", "day_number"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    topic_id: Mapped[int | None] = mapped_column(ForeignKey("topics.id"))
    day_number: Mapped[int] = mapped_column(Integer, nullable=False)
    minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    priority: Mapped[float] = mapped_column(Float, nullable=False, server_default=text("0"))
    status: Mapped[str] = mapped_column(String, nullable=False, server_default=text("'pending'"))
    source_chunk_ids: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'[]'")
    )
    is_retest: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))


class PlanSetting(Base):
    __tablename__ = "plan_settings"
    __table_args__ = (CheckConstraint("id = 1"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    total_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    days: Mapped[int] = mapped_column(Integer, nullable=False)


class DatabaseRow(Mapping[str, Any]):
    def __init__(self, row: Any):
        self._mapping = row._mapping
        self._values = tuple(row)

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return self._values[key]
        return self._mapping[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)


class DatabaseResult:
    def __init__(self, result: CursorResult[Any]):
        self._result = result

    @property
    def lastrowid(self) -> int | None:
        return self._result.lastrowid

    @property
    def rowcount(self) -> int:
        return self._result.rowcount

    def fetchone(self) -> DatabaseRow | None:
        row = self._result.fetchone()
        return DatabaseRow(row) if row is not None else None

    def fetchall(self) -> list[DatabaseRow]:
        return [DatabaseRow(row) for row in self._result.fetchall()]

    def __iter__(self) -> Iterator[DatabaseRow]:
        for row in self._result:
            yield DatabaseRow(row)


class DatabaseConnection:
    def __init__(self, connection: Connection):
        self._connection = connection

    def execute(self, statement: str, parameters: Any = ()) -> DatabaseResult:
        if isinstance(parameters, Mapping):
            bound_parameters = parameters
        else:
            index = 0

            def bind_parameter(_match: re.Match[str]) -> str:
                nonlocal index
                name = f"p{index}"
                index += 1
                return f":{name}"

            translated = re.sub(r"\?", bind_parameter, statement)
            if index != len(parameters):
                raise ValueError("SQL placeholder count does not match the parameters.")
            statement = translated
            bound_parameters = {
                f"p{parameter_index}": value
                for parameter_index, value in enumerate(parameters)
            }
        result = self._connection.execute(text(statement), bound_parameters)
        return DatabaseResult(result)


def create_database_engine(database_path: Path) -> Engine:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        URL.create("sqlite", database=str(database_path.resolve())),
        connect_args={"check_same_thread": False, "timeout": 30},
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection: Any, _connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("PRAGMA journal_mode = WAL")
        cursor.execute("PRAGMA busy_timeout = 30000")
        cursor.execute("PRAGMA synchronous = NORMAL")
        cursor.close()

    return engine


@contextmanager
def database_connection(engine: Engine) -> Iterator[DatabaseConnection]:
    with engine.begin() as connection:
        yield DatabaseConnection(connection)


def create_schema(engine: Engine) -> None:
    Base.metadata.create_all(engine)
