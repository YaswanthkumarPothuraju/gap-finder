import math
import re
from collections import Counter


STOP_WORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been",
    "before", "being", "between", "but", "can", "could", "does", "during",
    "each", "for", "from", "had", "has", "have", "into", "its", "more",
    "most", "not", "of", "on", "only", "other", "our", "out", "over",
    "same", "should", "some", "such", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "those", "through",
    "under", "use", "used", "using", "was", "were", "what", "when",
    "where", "which", "while", "with", "would", "you", "your",
}


def tokenize(text):
    return [
        word for word in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(word) > 2 and word not in STOP_WORDS
    ]


def chunk_text(text, max_words=400, overlap=50):
    """Split text by headings, then into overlapping word windows."""
    chunks = []
    section = "General"
    buffer = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        is_heading = line.startswith("#") or (
            len(line) < 100 and not re.search(r"[.!?]$", line)
            and bool(re.match(r"^(?:\d+(?:\.\d+)*[.)]?\s+|[A-Z][A-Z0-9 \-/]{2,})", line))
        )
        if is_heading and buffer:
            chunks.extend(_split_words(" ".join(buffer), section, max_words, overlap))
            buffer = []
        if is_heading:
            section = re.sub(r"^#+\s*|^\d+(?:\.\d+)*[.)]?\s*", "", line).strip() or line
        else:
            buffer.append(line)
    if buffer:
        chunks.extend(_split_words(" ".join(buffer), section, max_words, overlap))
    if not chunks and (text or "").strip():
        chunks.extend(_split_words((text or "").strip(), section, max_words, overlap))
    return chunks


def _split_words(text, section, max_words, overlap):
    words = text.split()
    result = []
    step = max(1, max_words - overlap)
    for start in range(0, len(words), step):
        body = " ".join(words[start:start + max_words]).strip()
        if body:
            result.append({"section": section, "text": body})
    return result


def parse_syllabus(text):
    """Extract a usable topic list from Markdown, numbered, or short headings."""
    return [node["title"] for node in parse_syllabus_tree(text)]


def parse_syllabus_tree(text):
    topics = []
    seen = set()
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        markdown = re.match(r"^(#+)\s*(.+)$", line)
        numbered = re.match(r"^(\d+(?:\.\d+)*)(?:[.)])?\s+(.+)$", line)
        heading = bool(markdown or numbered or re.match(
            r"^(?:unit|chapter|module)\s+\d+\b", line, re.I
        ))
        short_label = len(line) <= 80 and not re.search(r"[.!?]$", line)
        if not (heading or short_label):
            continue
        if markdown:
            depth = len(markdown.group(1))
            title = markdown.group(2)
        elif numbered:
            depth = len(numbered.group(1).split("."))
            title = numbered.group(2)
        else:
            depth = 1
            title = re.sub(
                r"^(?:unit|chapter|module)\s+\d+\s*[:.\-]?\s*",
                "",
                line,
                flags=re.I,
            )
        title = title.strip(" :-\t")
        key = title.casefold()
        if len(title) >= 3 and key not in seen:
            seen.add(key)
            topics.append({"title": title, "depth": depth})
    if not topics and (text or "").strip():
        topics = [{"title": re.sub(r"\s+", " ", (text or "").strip())[:80], "depth": 1}]
    return topics[:100]


def lexical_similarity(left, right):
    a = Counter(tokenize(left))
    b = Counter(tokenize(right))
    if not a or not b:
        return 0.0
    shared = set(a) & set(b)
    dot = sum(a[word] * b[word] for word in shared)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def cosine_similarity(left, right):
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(value * value for value in left))
    norm_right = math.sqrt(sum(value * value for value in right))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


def compute_mastery(attempts):
    """Return weighted mastery and answered-question count for one topic."""
    if not attempts:
        return None, 0
    latest = max(attempt["attempt_no"] for attempt in attempts)
    numerator = 0.0
    denominator = 0.0
    for attempt in attempts:
        recency = 0.7 ** (latest - attempt["attempt_no"])
        difficulty = {"easy": 1.0, "medium": 1.5, "hard": 2.0}.get(attempt["difficulty"], 1.0)
        weight = recency * difficulty
        numerator += weight * max(0.0, min(1.0, attempt["score"]))
        denominator += weight
    return (numerator / denominator if denominator else None), len(attempts)


def mastery_label(mastery, count):
    if mastery is None:
        return "needs more data"
    if count < 3:
        return "needs more data"
    if mastery < 0.5:
        return "weak"
    if mastery <= 0.75:
        return "medium"
    return "strong"


def topic_priority(mastery, exam_weight, is_prerequisite=False):
    score = 0.0 if mastery is None else (1.0 - mastery) * max(0.0, exam_weight)
    return score * (1.5 if is_prerequisite else 1.0)


def allocate_diagnostic_questions(topics, max_topics=30, max_questions=30, min_questions=10):
    """Give each selected leaf a question, then apportion extras by exam weight."""
    chosen = topics[:max_topics]
    if not chosen:
        return []
    target = min(max_questions, max(min_questions, len(chosen)))
    weights = {
        topic["id"]: max(0.0, float(topic.get("exam_weight", 1.0)))
        for topic in chosen
    }
    if not sum(weights.values()):
        weights = {topic["id"]: 1.0 for topic in chosen}
    total_weight = sum(weights.values())
    counts = {topic["id"]: 1 for topic in chosen}
    for _ in range(target - len(chosen)):
        next_topic = max(
            chosen,
            key=lambda topic: (
                target * weights[topic["id"]] / total_weight - counts[topic["id"]],
                -topic["id"],
            ),
        )
        counts[next_topic["id"]] += 1
    return [
        (topic, index)
        for topic in chosen
        for index in range(counts[topic["id"]])
    ]


def estimate_topic_minutes(word_count, mastery):
    reading_minutes = max(1, math.ceil(max(0, word_count) / 150))
    practice_minutes = 10
    weakness_factor = 1.0 + (1.0 - mastery) if mastery is not None else 1.25
    return max(5, math.ceil((reading_minutes + practice_minutes) * weakness_factor))


def allocate_plan(topics, total_minutes, days, retest_minutes=5):
    """Allocate a bounded plan, surfacing topics that do not fit."""
    total_minutes = max(0, int(total_minutes))
    days = max(1, int(days))
    daily_budgets = [total_minutes // days] * days
    for index in range(total_minutes % days):
        daily_budgets[index] += 1

    retest_budgets = [min(retest_minutes, budget) for budget in daily_budgets]
    study_budgets = [budget - retest for budget, retest in zip(daily_budgets, retest_budgets)]
    scheduled = []
    unscheduled = []
    day_index = 0
    for topic in topics:
        remaining = max(0, int(topic["minutes"]))
        while remaining and day_index < days:
            if study_budgets[day_index] == 0:
                day_index += 1
                continue
            allocated = min(remaining, study_budgets[day_index])
            scheduled.append({
                **topic,
                "day": day_index + 1,
                "minutes": allocated,
                "continued": allocated < remaining,
            })
            study_budgets[day_index] -= allocated
            remaining -= allocated
            if remaining:
                day_index += 1
        if remaining:
            unscheduled.append({**topic, "minutes": remaining})

    retests = [
        {"day": index + 1, "minutes": minutes}
        for index, minutes in enumerate(retest_budgets) if minutes > 0
    ]
    return scheduled, retests, unscheduled
