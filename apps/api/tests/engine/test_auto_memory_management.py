"""
Integration tests for automatic memory management in BaseEngine.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.memory_management import MemoryConfig


class TestBaseEngineAutoMemoryManagement:
    """Test that BaseEngine uses auto budget offloading by default."""

    def test_auto_memory_management_flag_default(self):
        """Test that auto_memory_management defaults to True."""
        from src.engine.base_engine import BaseEngine

        # Class variable should default to True
        assert BaseEngine.auto_memory_management is True

    def test_determine_memory_strategy_uses_auto_budget(self):
        """Test that _determine_memory_strategy returns auto budget config."""
        from src.engine.base_engine import BaseEngine

        # Create a minimal mock engine
        engine = MagicMock(spec=BaseEngine)
        engine.logger = MagicMock()
        engine.config = {"metadata": {"id": "test-model"}}

        # Create mocks that return actual values (not MagicMock)
        memory_info_mock = MagicMock(return_value={
            "gpu_total": 16.0,  # 16GB VRAM
            "gpu_available": 14.0,  # 14GB available
            "cpu_available": 32.0,  # 32GB RAM
        })

        component_size_mock = MagicMock(return_value=12 * 1024**3)  # 12GB model
        block_structure_mock = MagicMock(return_value=(100 * 1024**2, 10))  # 100MB blocks, 10 blocks

        # Apply patches
        engine._get_system_memory_info = memory_info_mock
        engine._estimate_component_model_size_bytes = component_size_mock
        engine._estimate_block_structure = block_structure_mock

        # Call the actual method
        component = {
            "type": "transformer",
            "name": "test_transformer"
        }
        config = BaseEngine._determine_memory_strategy(engine, component)

        # Verify it returns a MemoryConfig with budget offloading
        assert config is not None
        assert isinstance(config, MemoryConfig)
        assert config.offload_mode == "budget"
        assert config.budget_mb == "auto"
        assert config.async_transfers is True
        assert config.vram_safety_coefficient == 0.8

    def test_auto_memory_management_disabled_when_explicit_config(self):
        """Test that explicit user config overrides auto detection."""
        from src.engine.base_engine import BaseEngine

        engine = MagicMock(spec=BaseEngine)
        engine.logger = MagicMock()
        engine.auto_memory_management = True

        # User provides explicit config
        user_config = {
            "transformer": MemoryConfig(
                offload_mode="budget",
                budget_mb=3000  # Manual budget
            )
        }

        # Simulate the normalization logic
        with patch.object(
            BaseEngine,
            '_auto_memory_management_from_components',
            return_value={}
        ):
            normalized = BaseEngine._normalize_memory_management(
                engine,
                user_config,
                allow_auto=True  # Should be ignored because explicit config exists
            )

            # User's explicit config should be preserved
            assert "transformer" in normalized
            assert normalized["transformer"].budget_mb == 3000

    def test_memory_config_default_auto_budget(self):
        """Test that MemoryConfig defaults to auto budget."""
        config = MemoryConfig(offload_mode="budget")

        # Should default to "auto"
        assert config.budget_mb == "auto"

    def test_memory_config_manual_override(self):
        """Test that users can override auto with manual budget."""
        config = MemoryConfig(
            offload_mode="budget",
            budget_mb=5000
        )

        # Manual budget should be respected
        assert config.budget_mb == 5000

    def test_memory_config_disable_budget(self):
        """Test that users can disable budget offloading."""
        config = MemoryConfig(
            offload_mode="budget",
            budget_mb=None
        )

        # None should disable budgeting
        assert config.budget_mb is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
