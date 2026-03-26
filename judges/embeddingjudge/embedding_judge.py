import asyncio
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

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


class EmbeddingJudge:

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

        # Setup embedding model
        attn_implementation = "eager"  
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")

        # Collect all requests with metadata
        requests_info: List[Tuple[str, str, str, str]] = []  # (run_id, topic_id, request, response)
        for i, response in enumerate(rag_responses):
            query = topic_titles.get(response.metadata.topic_id, "")
            full_response = response.get_report_text().lower() if response.responses else ""

            requests_info.append((
                response.metadata.run_id,
                response.metadata.topic_id,
                query,
                full_response
            ))
                
        queries = [item[2] for item in requests_info]
        documents = [item[3] for item in requests_info]

        query_embeddings = model.encode(queries, batch_size=32, show_progress_bar=True)
        document_embeddings = model.encode(documents, batch_size=32, show_progress_bar=True)

        result_scores = np.sum(query_embeddings * document_embeddings, axis=1)

        builder = LeaderboardBuilder(KEYWORD_SPEC)
        
        for i, (run_id, topic_id, _, _) in enumerate(requests_info):
            score = float(result_scores[i])
            
            builder.add(
                run_id=run_id, 
                topic_id=topic_id, 
                values={"Relevance": score}
            )

        return builder.build(expected_topic_ids=expected_topic_ids, on_missing="fix_aggregate")
        