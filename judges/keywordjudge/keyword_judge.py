import asyncio
import re
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

KEYWORD_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("Relevance"),
))


class KeywordJudge:

    def judge(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        **kwargs: Any,
    ) -> Leaderboard:
        topic_titles: Dict[str, str] = {t.request_id: t.problem_statement or "" for t in rag_topics}
        expected_topic_ids: List[str] = list(topic_titles.keys())

        # Collect all requests with metadata
        requests_info: List[Tuple[str, str, MinimaLlmRequest]] = []  # (run_id, topic_id, request)
        for i, response in enumerate(rag_responses):
            query = topic_titles.get(response.metadata.topic_id, "")
            full_response = response.get_report_text().lower() if response.responses else ""
            #print(full_response)
            requests_info.append((
                response.metadata.run_id,
                response.metadata.topic_id,
                MinimaLlmRequest(
                    request_id=f"q{i}",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a relevance evaluator that evaluates the relevance between a query and a response.\n"
                                "Step 1: Generate exactly two paraphrases of the query.\n"
                                "Step 2: Score relevance between the query and response (0 to 3) for:\n"
                                "- Original query\n"
                                "- Query Paraphrase 1\n"
                                "- Query Paraphrase 2\n\n"
                                "Return ONLY three integers separated by spaces which represents the relevance between each query and the response.\n"
                                "Example: 2 1 3"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Query: {query}\n\n"
                                f"Response: {full_response}"
                            ),
                        },
                    ],
                    temperature=0.5,
                ),
            ))

        # Run all LLM requests in batch
        # Convert base config to full MinimaLlmConfig for backend features (batching, retry, etc.)
        full_config = MinimaLlmConfig.from_dict(llm_config.raw) if llm_config.raw else MinimaLlmConfig.from_env()
        backend = OpenAIMinimaLlm(full_config)
        llm_results = asyncio.run(backend.run_batched([req for _, _, req in requests_info]))
        print(llm_results)

        # Build leaderboard from responses
        builder = LeaderboardBuilder(KEYWORD_SPEC)
        for (run_id, topic_id, _), result in zip(requests_info, llm_results):
            scores = self._parse_relevance(result)
            avg_score = sum(scores) / 3
            builder.add(run_id=run_id, topic_id=topic_id, values={"Relevance": avg_score})

        return builder.build(expected_topic_ids=expected_topic_ids, on_missing="fix_aggregate")

    def _parse_relevance(self, result: Any) -> List[int]:
        if not isinstance(result, MinimaLlmResponse):
            print(f"[KeywordJudge] LLM error: {result}")
            return [0, 0, 0]

        text = result.text.strip().lower()
        numbers = re.findall(r"\b([0-3])\b", text)

        if len(numbers) >= 3:
            return [int(n) for n in numbers[:3]]

        return [0, 0, 0]
        