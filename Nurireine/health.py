"""
Health Check Module

Provides health monitoring for Nurireine bot components.
Can be queried to check if AI systems are loaded and operational.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class HealthChecker:
    """
    Monitors the health of bot components.
    
    Tracks:
    - AI system initialization status
    - Last successful operations
    - Error counts
    - Uptime
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        self.ai_loaded = False
        self.gatekeeper_healthy = False
        self.memory_healthy = False
        self.llm_healthy = False
        
        # Operation tracking
        self.last_analysis_time: Optional[datetime] = None
        self.last_response_time: Optional[datetime] = None
        self.last_db_operation: Optional[datetime] = None
        
        # Error tracking
        self.error_counts = {
            "analysis_errors": 0,
            "llm_errors": 0,
            "memory_errors": 0,
            "db_errors": 0
        }
        
        # Success tracking
        self.success_counts = {
            "analyses": 0,
            "responses": 0,
            "memory_operations": 0
        }
    
    def update_ai_status(
        self, 
        ai_loaded: bool, 
        gatekeeper_healthy: bool, 
        memory_healthy: bool, 
        llm_healthy: bool
    ):
        """Update AI system health status."""
        self.ai_loaded = ai_loaded
        self.gatekeeper_healthy = gatekeeper_healthy
        self.memory_healthy = memory_healthy
        self.llm_healthy = llm_healthy
    
    def record_analysis(self, success: bool = True):
        """Record an analysis attempt."""
        if success:
            self.success_counts["analyses"] += 1
            self.last_analysis_time = datetime.now()
        else:
            self.error_counts["analysis_errors"] += 1
    
    def record_response(self, success: bool = True):
        """Record a response generation attempt."""
        if success:
            self.success_counts["responses"] += 1
            self.last_response_time = datetime.now()
        else:
            self.error_counts["llm_errors"] += 1
    
    def record_memory_operation(self, success: bool = True):
        """Record a memory operation."""
        if success:
            self.success_counts["memory_operations"] += 1
        else:
            self.error_counts["memory_errors"] += 1
    
    def record_db_operation(self, success: bool = True):
        """Record a database operation."""
        if success:
            self.last_db_operation = datetime.now()
        else:
            self.error_counts["db_errors"] += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        # Determine overall health
        critical_systems = [
            self.ai_loaded,
            self.gatekeeper_healthy or self.llm_healthy  # At least one should work
        ]
        overall_healthy = all(critical_systems)
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "uptime_seconds": uptime,
            "ai_systems": {
                "loaded": self.ai_loaded,
                "gatekeeper": "healthy" if self.gatekeeper_healthy else "unavailable",
                "memory": "healthy" if self.memory_healthy else "unavailable",
                "llm": "healthy" if self.llm_healthy else "unavailable"
            },
            "last_operations": {
                "analysis": self.last_analysis_time.isoformat() if self.last_analysis_time else None,
                "response": self.last_response_time.isoformat() if self.last_response_time else None,
                "database": self.last_db_operation.isoformat() if self.last_db_operation else None
            },
            "statistics": {
                "success_counts": self.success_counts,
                "error_counts": self.error_counts,
                "total_operations": sum(self.success_counts.values()),
                "total_errors": sum(self.error_counts.values())
            }
        }
    
    def is_healthy(self) -> bool:
        """Quick health check."""
        return self.ai_loaded and (self.gatekeeper_healthy or self.llm_healthy)


# Global health checker instance
_health_checker = HealthChecker()


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    return _health_checker
