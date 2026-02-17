"""
Metrics and Observability Module

Provides in-memory metrics collection for performance monitoring.
Metrics are aggregated and reset every 24 hours.
All logged metrics use key=value format for easy parsing.
"""

import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class MetricsSnapshot:
    """Snapshot of current metrics."""
    timestamp: datetime
    
    # Response metrics
    total_responses: int = 0
    successful_responses: int = 0
    failed_responses: int = 0
    avg_response_latency_ms: float = 0.0
    p95_response_latency_ms: float = 0.0
    
    # Retrieval metrics
    total_retrievals: int = 0
    retrieval_hits: int = 0
    retrieval_misses: int = 0
    
    # Analysis metrics
    total_analyses: int = 0
    successful_analyses: int = 0
    failed_analyses: int = 0
    
    # Calculated rates
    failure_rate: float = 0.0
    retrieval_hit_rate: float = 0.0
    
    # Uptime
    uptime_seconds: float = 0.0


class MetricsCollector:
    """
    In-memory metrics collector with 24-hour auto-reset.
    
    Tracks:
    - Response latency (average and percentiles)
    - Failure rates (analysis, LLM, memory operations)
    - Retrieval hit rates (L3 memory hits vs misses)
    """
    
    def __init__(self, reset_hours: int = 24):
        """
        Initialize metrics collector.
        
        Args:
            reset_hours: Hours after which metrics are reset
        """
        self.reset_hours = reset_hours
        self.start_time = datetime.now()
        self.last_reset = datetime.now()
        
        # Response metrics
        self.response_latencies: deque = deque(maxlen=1000)  # Keep last 1000
        self.total_responses = 0
        self.successful_responses = 0
        self.failed_responses = 0
        
        # Retrieval metrics
        self.total_retrievals = 0
        self.retrieval_hits = 0
        self.retrieval_misses = 0
        
        # Analysis metrics
        self.total_analyses = 0
        self.successful_analyses = 0
        self.failed_analyses = 0
        
        logger.info(f"event=metrics_initialized reset_hours={reset_hours}")
    
    def _check_reset(self) -> None:
        """Check if metrics should be reset based on time."""
        now = datetime.now()
        if now - self.last_reset > timedelta(hours=self.reset_hours):
            logger.info(
                f"event=metrics_reset "
                f"total_responses={self.total_responses} "
                f"total_analyses={self.total_analyses} "
                f"total_retrievals={self.total_retrievals}"
            )
            self._reset()
    
    def _reset(self) -> None:
        """Reset all metrics."""
        self.last_reset = datetime.now()
        self.response_latencies.clear()
        self.total_responses = 0
        self.successful_responses = 0
        self.failed_responses = 0
        self.total_retrievals = 0
        self.retrieval_hits = 0
        self.retrieval_misses = 0
        self.total_analyses = 0
        self.successful_analyses = 0
        self.failed_analyses = 0
    
    def record_response(self, latency_ms: float, success: bool = True) -> None:
        """
        Record a response generation attempt.
        
        Args:
            latency_ms: Response latency in milliseconds
            success: Whether the response was successful
        """
        self._check_reset()
        
        self.total_responses += 1
        if success:
            self.successful_responses += 1
            self.response_latencies.append(latency_ms)
            logger.debug(
                f"event=response_recorded "
                f"latency_ms={latency_ms:.2f} "
                f"success=true"
            )
        else:
            self.failed_responses += 1
            logger.debug(
                f"event=response_recorded "
                f"latency_ms={latency_ms:.2f} "
                f"success=false"
            )
    
    def record_analysis(self, success: bool = True) -> None:
        """
        Record an analysis attempt.
        
        Args:
            success: Whether the analysis was successful
        """
        self._check_reset()
        
        self.total_analyses += 1
        if success:
            self.successful_analyses += 1
            logger.debug("event=analysis_recorded success=true")
        else:
            self.failed_analyses += 1
            logger.debug("event=analysis_recorded success=false")
    
    def record_retrieval(self, hit: bool = True) -> None:
        """
        Record a memory retrieval attempt.
        
        Args:
            hit: Whether the retrieval found relevant memories
        """
        self._check_reset()
        
        self.total_retrievals += 1
        if hit:
            self.retrieval_hits += 1
            logger.debug("event=retrieval_recorded hit=true")
        else:
            self.retrieval_misses += 1
            logger.debug("event=retrieval_recorded hit=false")
    
    def get_snapshot(self) -> MetricsSnapshot:
        """
        Get current metrics snapshot.
        
        Returns:
            MetricsSnapshot with current metrics
        """
        self._check_reset()
        
        # Calculate percentiles
        latencies = sorted(self.response_latencies)
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        p95_latency = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
        
        # Calculate rates
        failure_rate = (
            self.failed_responses / self.total_responses 
            if self.total_responses > 0 else 0.0
        )
        retrieval_hit_rate = (
            self.retrieval_hits / self.total_retrievals 
            if self.total_retrievals > 0 else 0.0
        )
        
        # Calculate uptime
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return MetricsSnapshot(
            timestamp=datetime.now(),
            total_responses=self.total_responses,
            successful_responses=self.successful_responses,
            failed_responses=self.failed_responses,
            avg_response_latency_ms=avg_latency,
            p95_response_latency_ms=p95_latency,
            total_retrievals=self.total_retrievals,
            retrieval_hits=self.retrieval_hits,
            retrieval_misses=self.retrieval_misses,
            total_analyses=self.total_analyses,
            successful_analyses=self.successful_analyses,
            failed_analyses=self.failed_analyses,
            failure_rate=failure_rate,
            retrieval_hit_rate=retrieval_hit_rate,
            uptime_seconds=uptime
        )
    
    def get_stats_dict(self) -> Dict:
        """
        Get metrics as a dictionary for display.
        
        Returns:
            Dictionary of current metrics
        """
        snapshot = self.get_snapshot()
        
        return {
            "timestamp": snapshot.timestamp.isoformat(),
            "uptime_hours": snapshot.uptime_seconds / 3600,
            "response_metrics": {
                "total": snapshot.total_responses,
                "successful": snapshot.successful_responses,
                "failed": snapshot.failed_responses,
                "failure_rate": f"{snapshot.failure_rate * 100:.2f}%",
                "avg_latency_ms": f"{snapshot.avg_response_latency_ms:.2f}",
                "p95_latency_ms": f"{snapshot.p95_response_latency_ms:.2f}"
            },
            "retrieval_metrics": {
                "total": snapshot.total_retrievals,
                "hits": snapshot.retrieval_hits,
                "misses": snapshot.retrieval_misses,
                "hit_rate": f"{snapshot.retrieval_hit_rate * 100:.2f}%"
            },
            "analysis_metrics": {
                "total": snapshot.total_analyses,
                "successful": snapshot.successful_analyses,
                "failed": snapshot.failed_analyses
            },
            "reset_info": {
                "last_reset": self.last_reset.isoformat(),
                "next_reset": (self.last_reset + timedelta(hours=self.reset_hours)).isoformat()
            }
        }


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """
    Get the global metrics collector instance.
    
    Returns:
        Global MetricsCollector instance
    """
    global _metrics_collector
    if _metrics_collector is None:
        import os
        reset_hours = int(os.getenv("METRICS_RESET_HOURS", "24"))
        _metrics_collector = MetricsCollector(reset_hours=reset_hours)
    return _metrics_collector


def record_response_metric(latency_ms: float, success: bool = True) -> None:
    """Convenience function to record response metric."""
    get_metrics_collector().record_response(latency_ms, success)


def record_analysis_metric(success: bool = True) -> None:
    """Convenience function to record analysis metric."""
    get_metrics_collector().record_analysis(success)


def record_retrieval_metric(hit: bool = True) -> None:
    """Convenience function to record retrieval metric."""
    get_metrics_collector().record_retrieval(hit)
