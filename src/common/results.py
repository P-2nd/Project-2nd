"""모델별 실행 결과를 단일 JSON 파일에 일관된 형식으로 저장한다."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import RESULT_DATA_PATH


SCHEMA_VERSION = 1


def roc_data_path(model_key: str, label: str) -> Path:
    """ROC 곡선에 쓸 Test 정답·확률 데이터의 저장 경로를 돌려준다."""
    return RESULT_DATA_PATH.parent / "roc" / f"{model_key}_{label}_auc_roc.parquet"


def upsert_result(
    *,
    model_key: str,
    model_name: str,
    label: str,
    experiment: dict[str, Any],
    metrics: dict[str, Any],
    threshold: float,
    total_time_sec: float,
    artifacts: dict[str, str],
    extras: dict[str, Any] | None = None,
) -> Path:
    """한 모델·실험 라벨의 최신 결과를 ``result_data.json``에 갱신한다.

    ``model_key``와 ``label`` 조합은 하나의 최신 실행 결과를 뜻한다. 같은 조합을
    다시 실행하면 해당 항목만 교체하므로, 서로 다른 모델의 기존 결과는 유지된다.
    """
    result_path = RESULT_DATA_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _load_result_data(result_path)

    record = {
        "model": model_name,
        "experiment": {"label": label, **experiment},
        "performance": {
            "total_time_sec": float(total_time_sec),
            "avg_latency_ms": float(metrics["average_inference_ms"]),
            "total_samples": int(metrics["total_samples"]),
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "auc_score": float(metrics["roc_auc"]),
        "threshold": float(threshold),
        "test_metrics": metrics,
        "artifacts": artifacts,
    }
    if extras:
        record.update(extras)
    payload.setdefault("results", {}).setdefault(model_key, {})[label] = record
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()

    temporary_path = result_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(result_path)
    return result_path


def _load_result_data(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "results": {}}

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
        raise ValueError(f"통합 결과 파일 형식이 올바르지 않습니다: {path}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"지원하지 않는 통합 결과 파일 버전입니다: {payload.get('schema_version')}"
        )
    return payload
