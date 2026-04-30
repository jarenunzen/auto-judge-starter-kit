import asyncio
import re
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from autojudge_base import (
    Leaderboard,
    LeaderboardBuilder,
    LeaderboardSpec,
    LlmConfigProtocol,
    MeasureSpec,
    NuggetBanksProtocol,
    Qrels,
    Report,
    Request,
)
from minima_llm import MinimaLlmConfig, MinimaLlmRequest, MinimaLlmResponse, OpenAIMinimaLlm

# ---------------------------------------------------------------------------
# Keyword injection helpers
# ---------------------------------------------------------------------------

# Common English stop words to exclude from keyword candidates
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "from", "up", "down", "out", "off", "over", "under",
    "again", "then", "once", "and", "but", "or", "nor", "so", "yet", "both",
    "either", "neither", "not", "only", "own", "same", "than", "too", "very",
    "just", "that", "this", "these", "those", "it", "its", "i", "me", "my",
    "we", "our", "you", "your", "he", "she", "they", "them", "his", "her",
    "their", "what", "which", "who", "whom", "how", "when", "where", "why",
}


def _extract_keywords(query: str, max_keywords: int = 10) -> List[str]:
    """Pull meaningful tokens out of the query string."""
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", query)
    return [t for t in tokens if t.lower() not in _STOP_WORDS][:max_keywords]


def _inject_keywords(
    response: str,
    keywords: List[str],
    num_injections: int = 3,
    seed: Optional[int] = None,
) -> str:
    """
    Randomly scatter *num_injections* keywords drawn from *keywords* into
    *response* at random word-boundary positions.  Falls back gracefully when
    the response is empty or the keyword list is shorter than requested.
    """
    if not keywords or not response.strip():
        return response

    rng = random.Random(seed)
    chosen = rng.choices(keywords, k=min(num_injections, len(keywords) * 3))

    # Split on whitespace so we can insert at word positions
    words = response.split()
    if not words:
        return " ".join(chosen)

    for kw in chosen:
        pos = rng.randint(0, len(words))
        words.insert(pos, kw)

    return " ".join(words)


KEYWORD_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("Relevance"),
    MeasureSpec("Confidence"),       
    MeasureSpec("WeightedRelevance"),
))

FOIL_SYSTEM_PROMPT = """\
You are a query disambiguation expert.
Given a query, produce ONE contrastive foil query: a plausible-sounding but \
semantically opposite or orthogonal information need that shares surface \
vocabulary with the original but seeks fundamentally different information.

Rules:
- Keep it short (under 15 words)
- It must share at least one keyword with the original
- It must be clearly seeking different information
- Output ONLY the foil query, nothing else\
"""


def make_foil_request(query: str, request_id: str) -> MinimaLlmRequest:
    return MinimaLlmRequest(
        request_id=request_id,
        messages=[
            {"role": "system", "content": FOIL_SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}"},
        ],
        temperature=0.7, 
    )


def make_static_negation(query: str) -> str:
    return f"Not about, related to, or relevant to the following: {query}"


JUDGE_SYSTEM_PROMPT = """\
You are a relevance evaluator.
You will be given a response and three queries.
Score the relevance of the response to each query independently on a scale of 0 to 3:
  0 = not relevant at all
  1 = marginally relevant
  2 = relevant
  3 = highly relevant

Return ONLY three integers separated by spaces, one per query, in order.
Example output: 3 1 0\
"""


def make_judge_request(
    response_text: str,
    query: str,
    contrastive_foil: str,
    lexical_negation: str,
    request_id: str,
) -> MinimaLlmRequest:
    return MinimaLlmRequest(
        request_id=request_id,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Response: {response_text}\n\n"
                    f"Query 1 (original):           {query}\n"
                    f"Query 2 (contrastive foil):   {contrastive_foil}\n"
                    f"Query 3 (lexical negation):   {lexical_negation}\n"
                ),
            },
        ],
        temperature=0.0,
    )


@dataclass
class JudgmentScores:
    s_original: int    
    s_contrastive: int  
    s_lexical: int      

    @property
    def is_reversed(self) -> bool:
        return self.s_contrastive > self.s_original or self.s_lexical > self.s_original

    @property
    def reversal_severity(self) -> float:
        excess_contrastive = max(0, self.s_contrastive - self.s_original)
        excess_lexical = max(0, self.s_lexical - self.s_original)
        return max(excess_contrastive, excess_lexical)

    @property
    def gap_contrastive(self) -> int:
        return self.s_original - self.s_contrastive

    @property
    def gap_lexical(self) -> int:
        return self.s_original - self.s_lexical

    @property
    def mean_gap(self) -> float:
        return (self.gap_contrastive + self.gap_lexical) / 2.0

    @property
    def confidence(self) -> float:
        if self.is_reversed:
            return 0.0
        return max(0.0, min(1.0, self.mean_gap / 3.0))

    @property
    def relevance(self) -> float:
        if self.is_reversed:
            return -self.reversal_severity
        return float(self.s_original)

    @property
    def weighted_relevance(self) -> float:
        if self.is_reversed:
            return self.relevance  
        return self.relevance * self.confidence

def parse_three_scores(result: Any) -> Tuple[int, int, int]:
    if not isinstance(result, MinimaLlmResponse):
        print(f"[NegationJudge] LLM error: {result}")
        return (0, 0, 0)
    numbers = re.findall(r"\b([0-3])\b", result.text.strip())
    if len(numbers) >= 3:
        return (int(numbers[0]), int(numbers[1]), int(numbers[2]))
    if len(numbers) == 2:
        return (int(numbers[0]), int(numbers[1]), int(numbers[1]))
    return (0, 0, 0)


def parse_foil(result: Any, fallback: str) -> str:
    if not isinstance(result, MinimaLlmResponse):
        return fallback
    text = result.text.strip()
    text = re.sub(r"^(foil|query|contrastive foil)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text if text else fallback


class NegationJudge:

    def __init__(
        self,
        inject_keywords: bool = True,
        num_injections: int = 25,
        injection_seed: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        inject_keywords:
            Master switch.  Set to False to reproduce the original behaviour.
        num_injections:
            How many keyword tokens to scatter into each response.
        injection_seed:
            Optional RNG seed for reproducibility (None = non-deterministic).
        """
        self.inject_keywords = inject_keywords
        self.num_injections = num_injections
        self.injection_seed = injection_seed

    def judge(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        **kwargs: Any,
    ) -> Leaderboard:
        topic_titles: Dict[str, str] = {
            t.request_id: t.problem_statement or "" for t in rag_topics
        }
        expected_topic_ids: List[str] = list(topic_titles.keys())

        full_config = (
            MinimaLlmConfig.from_dict(llm_config.raw)
            if llm_config.raw
            else MinimaLlmConfig.from_env()
        )
        backend = OpenAIMinimaLlm(full_config)

        unique_queries: Dict[str, str] = {}  
        responses_list = list(rag_responses)  

        for response in responses_list:
            tid = response.metadata.topic_id
            if tid not in unique_queries:
                unique_queries[tid] = topic_titles.get(tid, "")

        foil_requests: List[Tuple[str, MinimaLlmRequest]] = [
            (tid, make_foil_request(query, request_id=f"foil_{tid}"))
            for tid, query in unique_queries.items()
        ]
        foil_results = asyncio.run(
            backend.run_batched([req for _, req in foil_requests])
        )

        foils: Dict[str, str] = {}
        for (tid, _), result in zip(foil_requests, foil_results):
            query = unique_queries[tid]
            foils[tid] = parse_foil(result, fallback=make_static_negation(query))

        print("[NegationJudge] Generated foils:")
        for tid, foil in foils.items():
            print(f"  {tid}: {foil!r}")

        judge_meta: List[Tuple[str, str]] = [] 
        judge_requests: List[MinimaLlmRequest] = []

        for i, response in enumerate(responses_list):
            tid = response.metadata.topic_id
            query = unique_queries.get(tid, "")
            response_text = response.get_report_text().lower() if response.responses else ""

            if self.inject_keywords:
                keywords = _extract_keywords(query)
                seed = (self.injection_seed + i) if self.injection_seed is not None else None
                response_text = _inject_keywords(
                    response_text,
                    keywords,
                    num_injections=self.num_injections,
                    seed=seed,
                )

            judge_meta.append((response.metadata.run_id, tid))
            judge_requests.append(
                make_judge_request(
                    response_text=response_text,
                    query=query,
                    contrastive_foil=foils.get(tid, make_static_negation(query)),
                    lexical_negation=make_static_negation(query),
                    request_id=f"judge_{i}",
                )
            )

        judge_results = asyncio.run(backend.run_batched(judge_requests))
        print(f"[NegationJudge] Judge results: {judge_results}")

        builder = LeaderboardBuilder(KEYWORD_SPEC)

        for (run_id, topic_id), result in zip(judge_meta, judge_results):
            s_orig, s_cont, s_lex = parse_three_scores(result)
            scores = JudgmentScores(
                s_original=s_orig,
                s_contrastive=s_cont,
                s_lexical=s_lex,
            )

            if scores.is_reversed:
                print(
                    f"[NegationJudge] ASYMMETRY DETECTED — {run_id}/{topic_id}: "
                    f"orig={s_orig} contrastive={s_cont} lexical={s_lex} "
                    f"(severity={scores.reversal_severity:.1f})"
                )

            builder.add(
                run_id=run_id,
                topic_id=topic_id,
                values={
                    "Relevance": scores.relevance,
                    "Confidence": scores.confidence,
                    "WeightedRelevance": scores.weighted_relevance,
                },
            )

        return builder.build(
            expected_topic_ids=expected_topic_ids,
            on_missing="fix_aggregate",
        )